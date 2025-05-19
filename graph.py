import re
import matplotlib.pyplot as plt

# Read the new log file
with open("Test-cifar10/5748_EM_DARTS_Preventing_Perfo_Supplementary Material/em-darts/output/train/cifar10/Structure_train-20250512-102151/train.log", "r") as file:
    log_data = file.read()

# Extract training accuracy from Acc@1 averages at the end of each training epoch
train_acc = [float(m) for m in re.findall(r"Acc@1: [\d.]+ \(([\d.]+)\)", log_data)]

# Extract validation accuracy
valid_acc = [float(m) for m in re.findall(r"valid_acc ([\d.]+)", log_data)]

# Make sure lengths match (optional check)
epochs = list(range(min(len(train_acc), len(valid_acc))))
train_acc = train_acc[:len(epochs)]
valid_acc = valid_acc[:len(epochs)]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_acc, label="Training Accuracy", marker='o')
plt.plot(epochs, valid_acc, label="Validation Accuracy", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")  # Save to file
plt.show()
