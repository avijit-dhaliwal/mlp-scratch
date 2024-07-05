import subprocess
import json
import matplotlib.pyplot as plt

def run_implementation(language):
    if language == "cpp":
        cmd = ["./cpp/mlp"]
    elif language == "python":
        cmd = ["python", "python/main.py"]
    elif language == "go":
        cmd = ["go", "run", "go/main.go", "go/mlp.go", "go/data_loader.go"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {language} implementation:")
        print(e.stdout)
        print(e.stderr)
        return None
    except json.JSONDecodeError:
        print(f"Error parsing JSON from {language} implementation:")
        print(result.stdout)
        return None

def plot_results(results):
    languages = list(results.keys())
    
    # Training time comparison
    plt.figure(figsize=(10, 5))
    plt.bar(languages, [r['training_time'] for r in results.values()])
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.savefig('training_time_comparison.png')
    plt.close()

    # Final accuracy comparison
    plt.figure(figsize=(10, 5))
    plt.bar(languages, [r['accuracy'] for r in results.values()])
    plt.title('Final Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_comparison.png')
    plt.close()

    # Accuracy over epochs
    plt.figure(figsize=(10, 5))
    for lang, r in results.items():
        plt.plot(range(1, len(r['accuracies'])+1), r['accuracies'], label=lang)
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_over_epochs.png')
    plt.close()

if __name__ == "__main__":
    languages = ["cpp", "python", "go"]
    results = {}
    for lang in languages:
        print(f"Running {lang} implementation...")
        result = run_implementation(lang)
        if result:
            results[lang] = result
    
    if results:
        plot_results(results)
        print("Visualizations saved as PNG files.")
    else:
        print("No results to visualize.")