import subprocess

def run_experiment():
    # Command to run
    n = -1
    train_dir = ""
    test_dir = ""
    config_paths = ["config_deep1.yml"]
    for config_path in config_paths:
        try:
            with open(config_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('n '):
                        _, n = line.split() 
                    if line.startswith('train_dir '):
                        _, train_dir = line.split()
                    if line.startswith('test_dir '):
                        _, test_dir = line.split()
                    if line.startswith('dataset '):
                        _, dataset = line.split()
                        
                    
                if (n == -1):
                    raise ValueError("No line starting with 'n ' found in config file.")
        except Exception as e:
            print(f"Error reading config file: {e}")
            raise
        # n="10000"
        command = [
            "./build/experiment",
            train_dir,
            test_dir,
            dataset,
            n,
            config_path
        ]
        print('Command to run:', ' '.join(command))

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            print("Running command...\n")
            for line in iter(process.stdout.readline, ''):
                print(line, end='')

            process.wait()

            if process.returncode != 0:
                print("\nError occurred:")
                print(process.stderr.read())
            else:
                print("\nCommand executed successfully!")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    run_experiment()