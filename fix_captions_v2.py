import os
import re

mapping = {
    "38": "2-1",
    "39": "2-2",
    "40": "2-3",
    "41": "2-4",
    "26": "2-5",
    "12": "2-6",
    "43": "2-7",
    "50": "2-8",
    "47": "2-9",
    "48": "2-10",
    "49": "2-11",
    "51": "2-12",
    "52": "2-13",
    "44": "2-14",
    "45": "2-15",
    "101": "2-16",
    "53": "2-17"
}

images_dir = r"c:\Projects\LLM-Course\llm-course\static\images\week02"

for idx, new_num in mapping.items():
    txt_path = os.path.join(images_dir, f"{idx}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Replace the first occurrence of "Figure X-Y." with "Figure {new_num}."
        # We need a regex because we don't know X-Y precisely without looking it up again
        new_content = re.sub(r"^Figure \d+-\d+\.", f"Figure {new_num}.", content, count=1)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {txt_path} to Figure {new_num}.")
    else:
        print(f"Missing {txt_path}")
