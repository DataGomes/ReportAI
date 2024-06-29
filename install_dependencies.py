import sys
import subprocess
import os

def install_system_dependencies():
    if sys.platform.startswith('linux'):
        try:
            subprocess.check_call([
                'sudo', 'apt-get', 'update'
            ])
            subprocess.check_call([
                'sudo', 'apt-get', 'install', '-y',
                'build-essential', 'python3-dev', 'python3-pip', 'python3-setuptools',
                'python3-wheel', 'python3-cffi', 'libcairo2', 'libpango-1.0-0',
                'libpangocairo-1.0-0', 'libgdk-pixbuf2.0-0', 'libffi-dev', 'shared-mime-info'
            ])
            print("System dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install system dependencies automatically.")
            return False
    else:
        print("Automatic installation of system dependencies is only supported on Linux.")
        return False

def main():
    if install_system_dependencies():
        print("All system dependencies installed successfully.")
    else:
        print("\nPlease install the following system dependencies manually:")
        print("- build-essential")
        print("- python3-dev")
        print("- python3-pip")
        print("- python3-setuptools")
        print("- python3-wheel")
        print("- python3-cffi")
        print("- libcairo2")
        print("- libpango-1.0-0")
        print("- libpangocairo-1.0-0")
        print("- libgdk-pixbuf2.0-0")
        print("- libffi-dev")
        print("- shared-mime-info")
        print("\nOn Ubuntu/Debian, you can use the following command:")
        print("sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info")
        print("\nFor other operating systems, please refer to:")
        print("https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation")

if __name__ == "__main__":
    main()