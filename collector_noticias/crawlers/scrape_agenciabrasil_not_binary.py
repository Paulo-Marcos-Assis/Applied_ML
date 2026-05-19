#!/usr/bin/env python3
"""
Agência Brasil News Scraper
============================

Scrapes news articles from Agência Brasil /ultimas page.
Note: Agência Brasil doesn't have real pagination - all recent articles are on /ultimas page.
Saves to ../scraped/agenciabrasil/ directory.

Usage:
    python3 scrape_agenciabrasil.py
"""

import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import argparse
from datetime import datetime
import re

# Configuration
BASE_URL = "https://agenciabrasil.ebc.com.br/ultimas"
OUTPUT_DIR = Path(__file__).parent.parent / "scraped" / "agenciabrasil"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Rate limiting
DELAY_BETWEEN_ARTICLES = 0.5  # seconds


def scrape_article(url: str, session: requests.Session) -> dict:
    """
    Scrape a single article from Agência Brasil
    
    Args:
        url: Article URL
        session: Requests session
        
    Returns:
        Dictionary with article data or None if failed
    """
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else ''
        
        # Extract text from all paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20])
        
        # Extract date from URL pattern (e.g., /noticia/2026-05/...)
        date = ''
        url_date = re.search(r'/(\d{4}-\d{2})/', url)
        if url_date:
            date = url_date.group(1) + '-01'  # Approximate to first day of month
        
        if not title or not text:
            return None
        
        return {
            'url': url,
            'title': title,
            'text': text,
            'date': date,
            'portal': 'agenciabrasil',
            'scraped_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ✗ Error scraping {url}: {e}")
        return None


def scrape_page(session: requests.Session) -> list:
    """
    Scrape all article links from /ultimas page
    
    Args:
        session: Requests session
        
    Returns:
        List of article URLs
    """
    try:
        response = session.get(BASE_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find article links
        links = soup.find_all('a', href=True)
        article_links = []
        
        for a in links:
            href = a['href']
            
            # Convert relative URLs to absolute
            if href.startswith('/') and len(href) > 10 and href not in ['/', '/ultimas', '/fotos']:
                href = f'https://agenciabrasil.ebc.com.br{href}'
                article_links.append(href)
            # Include full URLs from same domain
            elif 'agenciabrasil.ebc.com.br' in href and href not in ['https://agenciabrasil.ebc.com.br/', 'https://agenciabrasil.ebc.com.br']:
                article_links.append(href)
        
        # Remove duplicates
        article_links = list(set(article_links))
        
        return article_links
        
    except Exception as e:
        print(f"  ✗ Error on /ultimas page: {e}")
        return []


def get_existing_urls() -> set:
    """
    Get URLs of already scraped articles
    
    Returns:
        Set of URLs
    """
    existing_urls = set()
    
    if OUTPUT_DIR.exists():
        for json_file in OUTPUT_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    existing_urls.add(article['url'])
            except:
                pass
    
    return existing_urls


def save_article(article: dict, index: int):
    """
    Save article to JSON file
    
    Args:
        article: Article dictionary
        index: File index number
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / f"agenciabrasil_{index:04d}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(article, f, ensure_ascii=False, indent=2)


def main():
    """
    Main scraping function
    
    Note: Agência Brasil doesn't have real pagination - all recent articles are on /ultimas page.
    """
    print("\n" + "="*80)
    print("AGÊNCIA BRASIL NEWS SCRAPER")
    print("="*80 + "\n")
    
    # Setup
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    
    # Get existing articles
    existing_urls = get_existing_urls()
    print(f"Already scraped: {len(existing_urls)} articles\n")
    
    print("Note: Agência Brasil shows all articles on /ultimas page (no pagination)\n")
    
    # Start scraping
    total_articles = len(existing_urls)
    new_articles = 0
    skipped = 0
    
    print("Scraping /ultimas page:\n")
    
    # Get article links from /ultimas page only
    article_urls = scrape_page(session)
    
    if not article_urls:
        print("  No articles found")
        return
    
    print(f"  Found {len(article_urls)} article links\n")
    
    # Scrape each article
    for url in article_urls:
        # Skip if already scraped
        if url in existing_urls:
            skipped += 1
            continue
        
        # Scrape article
        article = scrape_article(url, session)
        
        if article:
            total_articles += 1
            new_articles += 1
            save_article(article, total_articles)
            print(f"  ✓ [{total_articles}] {article['title'][:60]}...")
            existing_urls.add(url)
        
        time.sleep(DELAY_BETWEEN_ARTICLES)
    
    print(f"\n  Progress: {new_articles} new, {skipped} skipped\n")
    
    # Summary
    print("\n" + "="*80)
    print("SCRAPING COMPLETED")
    print("="*80)
    print(f"Total articles: {total_articles}")
    print(f"New articles: {new_articles}")
    print(f"Already existed: {len(existing_urls) - new_articles}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape Agência Brasil news articles')
    args = parser.parse_args()
    
    main()
