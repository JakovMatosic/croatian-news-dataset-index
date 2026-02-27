import pandas as pd
import requests
import json
from tqdm import tqdm
import concurrent.futures

# Load your data
INPUT_FILE = "index_hr_clean_vijesti_only.csv"
OUTPUT_FILE = "index_politics_final.csv"

df = pd.read_csv(INPUT_FILE)

def classify_row(row):
    text = f"{row['title']} {row['description']}"[:350] # Slightly longer context
    
    # This prompt defines the "hot" topics that split the public
    prompt = f"""Zadatak: Klasificiraj tekst kao 'DA' ako se bavi politikom ili polarizirajućim društvenim temama. 
U 'DA' spadaju:
- Politika (Vlada, Sabor, izbori, stranke, diplomacija).
- Ideologija (Ustaše, Partizani, NDH, komunizam, fašizam, Jasenovac, Bleiburg).
- Društvene debate (pobačaj/abortus, LGBT/Pride, vjeronauk, crkva u javnosti, Istanbulska konvencija).
- Nacionalne teme (Domovinski rat, branitelji, odnosi sa Srbijom, ćirilica).
- Svjetski sukobi (Rat u Ukrajini, Izrael/Gaza).

NE su:
- SPORT (olimpijadi, ski, tenis, nogomet, hokej, atletika, bilo koji sportski događaji ili sportaši poput Janice Kostelic, Federera, itd).
- Estrada, glazbeni tračevi, slavni ljudi bez političkog konteksta.
- Recepti, kulinarstvo.
- Crna kronika koja nema političkog aspekta.
- Kultura, film, knjiga bez politike.

Tekst: {text}
Odgovor (samo DA ili NE):"""

    try:
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 2} # Ultra-fast response
            }, timeout=7)
        answer = response.json().get('response', '').strip()
        answer_up = answer.upper()
        # Handle cases like "Da.", "DA, ovo je..." by checking the first token
        first_token = answer_up.split()[0].strip(".,:;!?\"'") if answer_up else ''
        label = "POLITIKA" if first_token.startswith("DA") else "NE"
        return label
    except Exception as e:
        return "ERROR"

def run_overnight():
    print(f"Starting classification of {len(df)} rows...")
    
    # We use 4 threads to keep the GPU busy without overwhelming the API
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(classify_row, df.to_dict('records')), total=len(df)))
    
    df['category'] = results
    
    # Filter and save
    politics_only = df[df['category'] == "POLITIKA"]
    politics_only.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"Success! Found {len(politics_only)} political articles.")

if __name__ == "__main__":
    run_overnight()