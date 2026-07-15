from re import S
from bs4 import BeautifulSoup
import requests

def fetch_site_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title else "Untitled"
    if soup.body:
        for irrelevant_tag in soup.body(["script", "style", "img", "input"]):
            irrelevant_tag.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = "No body found"
    return (title + "\n\n" + text)[:2_000]

def get_site_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.startswith("http"):
            links.append(href)
    return links