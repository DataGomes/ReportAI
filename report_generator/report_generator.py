import os
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from pybliometrics.scopus import ScopusSearch
import re
import markdown2
import logging
from together import Together
import pybliometrics
import voyageai
import nltk
import ftfy
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic import BERTopic
from hdbscan import HDBSCAN
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Attempt to import WeasyPrint, but handle the import error
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except OSError:
    WEASYPRINT_AVAILABLE = False
    print("-----")
    print("WeasyPrint could not import some external libraries. PDF generation will be disabled.")
    print("To enable PDF generation, please install the required system dependencies:")
    print("On Ubuntu/Debian:")
    print("sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info")
    print("For other operating systems, please refer to:")
    print("https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation")
    print("-----")

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportAI:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) - {'d'}
        self.punctuation_translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        self.vo = voyageai.Client()
        self.together_client = Together()
        self.counter = 0
        self.counter_lock = Lock()

    def clean(self, text):
        #remove @ppl, url
        output = re.sub(r'https://\S*','', text)
        output = re.sub(r'@\S*','',output)
        
        #remove \r, \n
        rep = r'|'.join((r'\r',r'\n'))
        output = re.sub(rep,'',output)
        
        #remove extra space
        output = re.sub(r'\s+', ' ', output).strip()

        return output

    def remove_punctuation(self, text: str) -> str:
        return text.translate(self.punctuation_translator)

    def format_query(self, terms: List[str]) -> str:
        formatted_terms = [f'TITLE-ABS ( "{term[:-1]}*" )' if term.endswith('s') else f'TITLE-ABS ( "{term}" )' for term in terms]
        return ' AND '.join(formatted_terms)
    
    def replace_quotes(self, strings):
        replaced_strings = [string.replace("'", "").replace('"', '') for string in strings]
        return replaced_strings

    def delete_brackets_content(self, lst):
        result = []
        for item in lst:
            if isinstance(item, str):
                item = re.sub(r'\[.*?\]', '', item)
                item = re.sub(r'\{.*?\}', '', item)
            result.append(item)
        return result
    
    def filter_df(self, df_scopus: pd.DataFrame) -> pd.DataFrame:
        # Filter out rows with invalid descriptions or DOIs
        mask_desc = df_scopus["description"].isna() | (df_scopus["description"] == '') | (df_scopus["description"] == '[No abstract available]')
        mask_doi = df_scopus["doi"].isnull() | (df_scopus["doi"] == '') | df_scopus['doi'].isna() | (df_scopus["doi"] == 'None')
        filtered_df = df_scopus[~(mask_desc | mask_doi)].reset_index(drop=True)

        # Clean and fix text in description and title
        for col in ['description', 'title']:
            filtered_df[col] = filtered_df[col].apply(lambda x: ftfy.fix_text(self.clean(x)))

        # Process abstracts
        abstracts = filtered_df['description'].tolist()
        abstracts = self.replace_quotes(self.delete_brackets_content(abstracts))

        all_sentences = []
        for idx, text in enumerate(abstracts):
            sentences = sent_tokenize(text)
            title = filtered_df['title'][idx]
            
            if len(sentences) == 1:
                all_sentences.append((f"{title}. {sentences[0]}", idx))
            elif len(sentences) == 2:
                all_sentences.append((f"{title}. {' '.join(sentences)}", idx))
            elif len(sentences) >= 3:
                for i in range(1, len(sentences) - 1):
                    all_sentences.append((f"{title}. {' '.join(sentences[i-1:i+2])}", idx))

        # Create a DataFrame with processed sentences
        df_sentences = pd.DataFrame(all_sentences, columns=['Text Response', 'Abstract Index'])
        df_sentences['sentence_index'] = df_sentences.index
        df_sentences.set_index("Abstract Index", inplace=True)

        # Merge the original DataFrame with the processed sentences
        result_df = filtered_df.merge(df_sentences, left_index=True, right_index=True, how='left')

        return result_df
    
    def refine_search(self, query: str, basis: str) -> Tuple[pd.DataFrame, Optional[str]]:
        max_attempts = 5
        year = 2019
        q = query + basis
        failure = None
        df_scopus = pd.DataFrame(columns=['title', 'description', 'doi'])

        for attempt in range(max_attempts):
            s = ScopusSearch(q, download=False)
            results_size = s.get_results_size()

            if results_size > 10000:
                year += 1
                basis = f' AND PUBYEAR > {year} AND LANGUAGE ("English") AND (DOCTYPE ("ar") OR DOCTYPE ("re") OR DOCTYPE ("cp"))'
                q = query + basis
                logger.info(f'Refining search for year > {year}, results now: {results_size}')
            elif results_size > 30:
                logger.info('Optimal number of results found.')
                s = ScopusSearch(q, verbose=True, view="COMPLETE")
                df_scopus = pd.DataFrame(s.results)
                break
        else:
            failure = 'Too many results, make the search more specific' if results_size > 10000 else 'Too few results, make the search less specific'

        logger.info(f'Final search results size: {results_size}')
        return df_scopus, failure

    def get_embeddings(self, texts, batch_size: int = 128, input_type: Optional[str] = None):
        
        if not isinstance(texts, list):
            texts = [texts]        
        texts = [text.replace("\n", " ") for text in texts]

        texts = ["Cluster the text: " + text for text in texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.vo.embed(batch_texts, model="voyage-large-2-instruct", input_type=input_type).embeddings
            all_embeddings.extend(batch_embeddings)

        if len(all_embeddings) == 1:
            return all_embeddings[0]  # This ensures the output is a single vector (1024,)
        else:
            return all_embeddings

    def search_embeddings(self, df: pd.DataFrame, theme: str, n: int = 100) -> pd.DataFrame:
        embedding = self.get_embeddings(theme)

        similarity_scores = []

        for idx,r in enumerate(df.Embedding):
            similarity = np.dot(embedding, r)
            similarity_scores.append(similarity)

        df['similarities'] = similarity_scores
        res = df.sort_values('similarities', ascending=False).head(n)
        return res
    
    def run_bertopic(self, data_embedding):
        embeddings = data_embedding["embedding"].tolist()
        docs = data_embedding["input"].tolist()
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
        vo = voyageai.Client()

        max_iterations = 50

        for i in range(max_iterations):
            random_state = 42 + i  # Increment random_state for each iteration

            umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', low_memory=False)
            umap_embeddings = umap_model.fit_transform(embeddings)

            ClusterSize = int(len(docs)/100)
            if ClusterSize < 10:
                ClusterSize = 10

            SampleSize = int(ClusterSize/10)
            if SampleSize < 5:
                SampleSize = 5

            hdbscan_model = HDBSCAN(gen_min_span_tree=True, prediction_data=True, min_cluster_size=ClusterSize,
                                    min_samples=SampleSize, metric='euclidean', cluster_selection_method='eom')
            vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3))
            empty_dimensionality_model = BaseDimensionalityReduction()

            topic_model = BERTopic(vectorizer_model=vectorizer_model, umap_model=empty_dimensionality_model,
                                hdbscan_model=hdbscan_model, top_n_words=10,
                                verbose=True, n_gram_range=(1, 3))
            topics = topic_model.fit_transform(docs, umap_embeddings)
            bertopic_df = topic_model.get_topic_info()

            # Check if the condition is met
            if len(bertopic_df) > 15:
                break  # Condition met, exit loop
            else:
                random_state += 1  # Increment random_state and continue loop
        return bertopic_df, umap_embeddings, topic_model
    
    def filter_bertopic(self, bertopic_df, query):
        client = Together()
        
        Keywords = bertopic_df['Representation']
        representative_docs = bertopic_df['Representative_Docs']

        formatted_keywords = [', '.join(sublist) for sublist in Keywords]
        formatted_docs = ['\n'.join(sublist) for sublist in representative_docs]

        responses = []
        keywords_list = []
        documents_list = []
        similarities_list = []
        index_list = []
        index_number = 0

        for i in range(len(formatted_keywords)):
            prompt = f"""
            You will receive keywords and a small sample of parts of documents from a topic. Assign a short label to the topic, based on the keywords. DO NOT make up new information that is not contained in the keywords.

            Example:
            Keywords:
            [keyword 1,keyword 2,keyword 3,keyword 4,keyword 5,keyword 6,keyword 7,keyword 8,keyword 9,keyword 10]

            Documents:
            [document 1.
            document 2.
            document 3]

            Topic assignment:
            [short topic based upon the keywords]
            
            INSTRUCTIONS:
            
            1. The answer should NOT use the word "topic" or "label".

            2. The label should have no more than 10 words, reflecting ONLY ONE topic.

            3. The label should be based ONLY on the keywords provided

            4. The answer should NOT be based on the documents, use only the keywords.
            
            4. Your answer should be short and succinctly reflect the main topic present ONLY on the Keywords.

            5. Prioritize the FIRST KEYWORDS more heavily when determining the answer, giving them greater weight than subsequent keywords

            Your task:
            Keywords: 
            [{formatted_keywords[i]}]

            Documents: 
            [{formatted_docs[i]}]

            Your response:
            """

            response = client.chat.completions.create(
                messages=[
                    {'role': 'system', 'content': 'You are an expert at creating topic labels. In this task, you will be provided with a set of keywords related to ONE particular topic. Your job is to use these keywords, prioritizing the first keywords, to come up with an accurate and short label for the topic. It is crucial that you base your label STRICLY on the keywords.'},
                    {'role': 'user', 'content': prompt},
                ],
                model="meta-llama/Llama-3-70b-chat-hf",
                temperature=0,
            )
            response_content = response.choices[0].message.content

            query_embedding = self.get_embeddings(query)


            #dimensionality of query_embedding print

            response_embedding = self.get_embeddings(response_content)

            similarity = np.dot(query_embedding, response_embedding)

            embeddings_df = pd.DataFrame({'Embedding': [response_embedding]})

            
            # Calculate similarity using dot product
            similarity_scores = []
            for idx,r in enumerate(embeddings_df.Embedding):
                similarity = np.dot(query_embedding, r)
                similarity_scores.append(similarity)
            #similarity = np.dot(query_embedding, response_embedding)

            responses.append(response_content)
            keywords_list.append(formatted_keywords[i])
            documents_list.append(formatted_docs[i])
            similarities_list.append(similarity)
            index_list.append(index_number)

            index_number += 1 

        topics = pd.DataFrame({
            "Response": responses,
            "Keywords": keywords_list,
            "Documents": documents_list,
            "Similarities": similarities_list,
            "Indexes": index_list
        })
        topics['Choice'] = 'Y'

        topics = topics.sort_values(by="Similarities", ascending=False)

        for index, row in topics.iterrows():
            if row['Similarities'] < 0.85:
                topics.at[index, 'Choice'] = 'N'

        n_indices = topics[topics['Choice'] == 'N'].index

        for idx in n_indices:
            current_min = topics[topics["Choice"] == 'Y']["Similarities"].min()
            if current_min > 0.85:
                current_min = 0.85
            if current_min - topics.loc[idx, "Similarities"] <= 0.01 and topics.loc[idx, "Similarities"] > 0.8:
                topics.loc[idx, "Choice"] = "Y"

        topics = topics.sort_values(by="Indexes", ascending=True)

        modified_topics = topics.drop(index=0)
        duplicates = modified_topics['Response'].duplicated(keep='first')
        duplicates = duplicates.reindex(topics.index, fill_value=False)
        topics.loc[duplicates, 'Choice'] = 'N'
        topics['Choice'][0] = 'N'
        return topics

    def generate_report(self, theme: str, df_scopus: pd.DataFrame, data_embedding) -> str:


        with self.counter_lock:
            self.counter += 1
            local_counter = self.counter

        data_embedding.rename(columns={'embedding': 'Embedding'}, inplace=True)
        res = self.search_embeddings(data_embedding, theme, n=100)
        index_list = res.index.tolist()
        filtered_result_df = df_scopus[df_scopus["sentence_index"].isin(index_list)]
        filtered_result_df['sentence_index'] = pd.Categorical(filtered_result_df['sentence_index'], categories=index_list, ordered=True)
        filtered_result_df = filtered_result_df.merge(res[['similarities']], left_on='sentence_index', right_index=True, how='left')
        filtered_result_df = filtered_result_df.sort_values(by='similarities', ascending=False)
        # Separate duplicates and non-duplicates
        duplicates = filtered_result_df[filtered_result_df.index.duplicated(keep=False)]
        non_duplicates = filtered_result_df[~filtered_result_df.index.duplicated(keep=False)]

        top_two_list = []

        if not duplicates.empty:
            for name, group in duplicates.groupby(duplicates.index):
                top_two_list.append(group.head(1))    

        top_two_duplicates = pd.concat(top_two_list) if top_two_list else pd.DataFrame()

        # Combine the top two duplicates with all non-duplicates
        final_result = pd.concat([non_duplicates, top_two_duplicates])

        # Sort by similarities in descending order
        filtered_result_df = final_result.sort_values(by='similarities', ascending=False)

        query = f"Is this text related to the topic: \"{theme}\"?"
        results = self.vo.rerank(query, filtered_result_df['description'].tolist(), model="rerank-1", top_k=3)

        textsranked = [item.document for item in results.results]
        formatted_top = '\n\n'.join(f"{doc}" for doc in textsranked)

        query = self._construct_query(theme, formatted_top)
        the_answer = self._get_ai_summary(query, theme)

        exact_matches = filtered_result_df[filtered_result_df['description'].isin(textsranked)]
        de_exact_matches = exact_matches.drop_duplicates(subset='doi', keep='first')

        doi_string = '\n\n'.join(
            f'- [{row["title"] or row["doi"]}](https://doi.org/{row["doi"]})'
            for index, row in de_exact_matches.iterrows()
        )

        return f"### {local_counter}. {theme.capitalize()}\n\n{the_answer}\n\n#### References\n{doi_string}"

    def _construct_query(self, theme: str, formatted_abstracts: str) -> str:
        return f"""You will receive a selection of parts of abstracts. Your task is to create a general summary of the topic:'{theme}' based ONLY upon the response of the selected abstracts.

        EXAMPLE:
        \"\"\"
        Text of abstract 1.

        Text of abstract 2.

        Text of abstract 3.

        YOUR RESPONSE:
        The {theme} Use text of abstract 1 to discuss the topic. Use text of abstract 2 to discuss the topic. Use text of abstract 3 to discuss the topic.
        \"\"\"

        INSTRUCTIONS:
        1. DO NOT use information outside of the provided text.
        2. The summary should be written in one paragraph.
        3. Create a summary based ONLY on the response. 
        4. Your answer should be a summary about the topic.
        5. Answer directly, DO NOT tell this is the summary or use the word "summary" or "abstract" in your response.
        6. Start your response mentioning {theme}.

        SELECTED ABSTRACTS:
        \"\"\"
        {formatted_abstracts}
        \"\"\"

        Your task is to create a general summary of the topic:'{theme}' based ONLY upon the abstracts.

        YOUR RESPONSE:
        """

    def _get_ai_summary(self, query: str, theme: str) -> str:
        response = self.together_client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': f'You are a scientist working on a project about scientific abstracts related to the topic: {theme}. You are an expert at writing summaries based upon abstracts. You DO NOT use information outside of the provided text. You are a scientific abstract summarizer, so the summary must be based ONLY on the provided text. Answer directly about the topic: {theme}, DO NOT tell this is the summary or use the word "summary" or "abstract" in your response.'},
                {'role': 'user', 'content': query},
            ],
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
        )
        return response.choices[0].message.content
    def create_markdown_toc(self, text):
        lines = text.split('\n')
        toc = ["## Table of topics"]
        
        for line in lines:
            if line.startswith('### '):
                title = line[4:].strip()
                # Use a more robust way to create slug for the link
                link = re.sub(r'[^\w\s-]', '', title).strip().lower().replace(' ', '-')
                toc.append(f"\n\n[{title}](#{link})")
        
        return '\n'.join(toc)

    def run_report(self, input_user: str, output_dir: str = None) -> Tuple[str, str, str]:
        clean_string = self.remove_punctuation(input_user).lower()
        words = word_tokenize(clean_string)
        terms = [word for word in words if word.lower() not in self.stop_words]

        query = self.format_query(terms)
        basis = ' AND PUBYEAR > 2019 AND LANGUAGE ( "English" ) AND (DOCTYPE ( "ar" ) OR DOCTYPE ( "re" ) OR DOCTYPE ( "cp" ))'

        df_scopus, failure = self.refine_search(query, basis)

        if failure:
            return failure, query, "failure"
        
        filtered_df = self.filter_df(df_scopus)

        # Get the embeddings
        embeddings = self.get_embeddings(filtered_df["Text Response"].tolist())

        # Create a DataFrame with both embeddings and input phrases
        data_embedding = pd.DataFrame({
            "embedding": embeddings,
            "input": filtered_df["Text Response"].tolist()
        })

        send_df, umap_embeddings, bertopic_model = self.run_bertopic(data_embedding)

        filtered_send_df = self.filter_bertopic(send_df, input_user)


        # Using ThreadPoolExecutor to parallelize API calls
        partial_report = partial(self.generate_report,  df_scopus=filtered_df, data_embedding = data_embedding)
        theme = filtered_send_df[filtered_send_df['Choice'] == 'Y']['Response'].tolist()
        with ThreadPoolExecutor() as executor:
            # Using map to ensure the order of results matches the order of 'themes'
            results = list(executor.map(partial_report, theme))
        # Combining all individual reports into a single string
        final_combined_report = "\n\n".join(results)

        TOC = self.create_markdown_toc(final_combined_report)
        inputpdf = pd.Series([input_user]).str.capitalize().values[0]
        introduction = f"This report provides an AI-based analysis of the most representative topics related to {input_user}, identified based on the search criteria. The references were directly extracted from scientific databases, while the summaries were constructed based upon the abstracts of the references using AI. By leveraging an extensive database of scientific sources, the report delivers reference-based results. While this report is based on scientific data sources, users should exercise caution in interpretation, given the inherent complexities and evolving nature of AI-based analysis."

        final_result_send =f"# {inputpdf}" + "\n\n" + "## Introduction" + "\n\n" + introduction + "\n\n" + TOC + "\n\n" + final_combined_report
        html_output = markdown2.markdown(final_result_send, extras=["toc", "headers-ids"])
        #report = self.generate_report(input_user, df_scopus)
        #html_output = markdown2.markdown(final_combined_report)

        if output_dir and WEASYPRINT_AVAILABLE:
            querypdf = pd.Series([input_user]).str.capitalize().values[0]
            query_path = re.sub(r'[^\w\s-]', '', query).replace(' ', '')
            
            absolute_path = os.path.join(output_dir, query_path)
            os.makedirs(absolute_path, exist_ok=True)
            
            model_save_path = os.path.join(absolute_path, f"{querypdf}.pdf")
            HTML(string=html_output).write_pdf(model_save_path)
            
            logger.info(f"Report generated. PDF saved to {model_save_path}")
        elif output_dir and not WEASYPRINT_AVAILABLE:
            logger.warning("PDF generation is disabled due to missing system dependencies.")
            # Save as HTML instead
            querypdf = pd.Series([input_user]).str.capitalize().values[0]
            query_path = re.sub(r'[^\w\s-]', '', query).replace(' ', '')
            
            absolute_path = os.path.join(output_dir, query_path)
            os.makedirs(absolute_path, exist_ok=True)
            
            model_save_path = os.path.join(absolute_path, f"{querypdf}.html")
            with open(model_save_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
            
            logger.info(f"Report generated. HTML saved to {model_save_path}")

        return html_output, query, html_output