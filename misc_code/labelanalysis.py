filename = "combo-labels.txt"

prominent_counts = {}
description_counts = {}

with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        parts = line.split("||")
        name = parts[0].strip()
        prominent_features = [f.strip() for f in parts[1].strip().split("|")]
        descriptions = [d.strip() for d in parts[2].strip().split("|")]
        descriptions2 = [d.split(",") for d in descriptions]
        for i, feature in enumerate(prominent_features):
            if feature not in prominent_counts:
                prominent_counts[feature] = set()
                description_counts[feature] = {}
            if descriptions[i]:
                description_counts[feature][descriptions[i]] = description_counts[feature].get(descriptions[i], 0) + 1
            prominent_counts[feature].add(name)

print("Number of prominent features:", len(prominent_counts.items()))
print("\nNumber of people with each prominent feature:")
for feature, count in prominent_counts.items():
    print(f"{feature}: {len(count)}")

print("\nDescription counts by feature:")
for feature, counts in description_counts.items():
    print(f"{feature}: {len(counts)}")
    for description, count in counts.items():
        print(f"  {description}: {count}")

