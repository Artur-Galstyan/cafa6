import requests


def get_uniprot_function(protein_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}?fields=cc_function,cc_subcellular_location,keyword"
    response = requests.get(url, headers={"Accept": "application/json"})
    if response.status_code == 200:
        data = response.json()
        for k, v in data.items():
            print(f"{k}={v}")
    return ""


if __name__ == "__main__":
    get_uniprot_function("P48547")
