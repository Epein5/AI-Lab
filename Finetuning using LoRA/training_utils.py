import torch
from tqdm import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train and validate a PyTorch model
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimization algorithm
    num_epochs : int
        Number of training epochs
    device : torch.device
        Device to run training on (CPU/GPU)
    
    Returns:
    --------
    dict: Training history with loss and accuracy metrics
    """

    # Initialize training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item()
        
        # Calculate metrics
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        # Store metrics
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        
        # Print metrics
        print(f'Epoch [{epoch+1}/{num_epochs}] | Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f} ')        
    
    print('Training completed!')
    
    return training_history

def plot_training_history(training_history):
    """
    Visualize training and validation metrics
    
    Parameters:
    -----------
    training_history : dict
        Dictionary containing training metrics
    """
    import matplotlib.pyplot as plt
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(training_history['train_loss'], label='Training Loss')
    ax1.plot(training_history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot validation accuracy
    ax2.plot(training_history['val_accuracy'], label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()