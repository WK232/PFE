import re
import matplotlib.pyplot as plt

log_file_path = "/home/kharratw/Documents/tessssst/PFE/EM-DARTS/output/train/cifar10/Structure_train-20250520-094533/train.log"

# Patterns for lines to extract
train_line_pattern = re.compile(r"Train:.*Acc@1:.*\((\d+\.\d+)\)")
valid_line_pattern = re.compile(r"Valid:.*Acc@1:.*\((\d+\.\d+)\)")
valid_acc_pattern = re.compile(r"^valid_acc\s*[:=]\s*(\d+\.\d+)", re.IGNORECASE)

# Containers for the last entries per epoch
train_acc = []
val_acc = []
valid_acc_lines = []

last_train = None
last_val = None

# Go through the log lines
with open(log_file_path, "r") as f:
    for line in f:
        # Track last train acc@1 until a validation line or new epoch comes
        train_match = train_line_pattern.search(line)
        if train_match:
            last_train = float(train_match.group(1))

        # When validation line appears, store the last recorded training and validation values
        val_match = valid_line_pattern.search(line)
        if val_match:
            last_val = float(val_match.group(1))
            if last_train is not None:
                train_acc.append(last_train)
                val_acc.append(last_val)
                last_train = None  # reset for next epoch

        # Collect valid_acc lines (if present)
        valid_acc_line = valid_acc_pattern.search(line)
        if valid_acc_line:
            valid_acc_lines.append(float(valid_acc_line.group(1)))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label='Train Accuracy (last per epoch)', marker='o')
plt.plot(val_acc, label='Validation Accuracy (last per epoch)', marker='s')
if valid_acc_lines:
    plt.plot(valid_acc_lines, label='valid_acc lines (last value)', marker='^')

plt.title('Final Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save to file
output_file = "final_accuracy_plot.png"
plt.savefig(output_file)
plt.show()

print(f"Plot saved as: {output_file}")
