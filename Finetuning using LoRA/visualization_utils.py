import matplotlib.pyplot as plt
import torch

def plot_class_accuracies(class_labels, accuracies):
    """
    Create a bar plot of accuracies for different classes
    
    Parameters:
    -----------
    class_labels : list
        List of class names
    accuracies : list
        List of accuracy percentages corresponding to class_labels
    """
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Shorten and clean up labels
    shortened_labels = [label.split('___')[-1].replace('_', ' ') for label in class_labels]
    
    # Create bars
    bars = plt.bar(shortened_labels, accuracies)
    
    # Set title and labels
    plt.title('Accuracy by Tomato Disease Class', fontsize=14, pad=20)
    plt.xlabel('Disease Classes', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%',
                ha='center', va='bottom')
    
    # Add grid and adjust layout
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(accuracies) + 5)  # Add some padding at the top
    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_loader, class_labels, device):
    """
    Evaluate model performance and calculate class-wise accuracies
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained PyTorch model
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    class_labels : list
        List of class names
    device : torch.device
        Device to run the evaluation on
    
    Returns:
    --------
    tuple: (overall_accuracy, class_accuracies, wrong_counts)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Initialize counters
    correct = 0
    total = 0
    wrong_counts = [0 for _ in range(len(class_labels))]
    class_accuracies = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for data in val_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            outputs = model(x)
            
            for idx, output in enumerate(outputs):
                predicted = torch.argmax(output)
                if predicted == y[idx]:
                    correct += 1
                else:
                    wrong_counts[y[idx].item()] += 1
                total += 1
    
    # Calculate overall accuracy
    overall_accuracy = round(correct/total, 3)
    print(f'Overall Accuracy: {overall_accuracy}')
    
    # Calculate and print class-wise accuracies
    for i in range(len(wrong_counts)):
        print(f'Wrong predictions for class {class_labels[i]}: {wrong_counts[i]}')
        class_total = sum(1 for _, label in val_loader.dataset if label == i)
        class_accuracy = round((class_total - wrong_counts[i]) / class_total, 3)
        class_accuracies.append(int(class_accuracy*100))
        print(f'Class {class_labels[i]} Accuracy: {class_accuracy}')
    
    return overall_accuracy, class_accuracies, wrong_counts