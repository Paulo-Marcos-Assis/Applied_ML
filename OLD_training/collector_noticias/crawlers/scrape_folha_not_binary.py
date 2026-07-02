#!/usr/bin/env python3
"""
Folha de S.Paulo News Scraper
==============================

Scrapes news articles from Folha de S.Paulo homepage.
Note: Folha doesn't have real pagination - all recent articles are on the homepage.
Saves to ../scraped/folha/ directory.

Usage:
    python3 scrape_folha.py
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
BASE_URL = "https://www1.folha.uol.com.br/ultimas-noticias/?p={}"
PAGE_1_URL = "https://www1.folha.uol.com.br/ultimas-noticias/"
OUTPUT_DIR = Path(__file__).parent.parent / "scraped" / "folha"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Rate limiting
DELAY_BETWEEN_ARTICLES = 0.5  # seconds
DELAY_BETWEEN_PAGES = 1.0     # seconds


def scrape_article(url: str, session: requests.Session) -> dict:
    """
    Scrape a single article from Folha de S.Paulo
    
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
        text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Extract date from URL pattern (YYYY/MM/DD) or from time tag
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
        if date_match:
            date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        else:
            # Try to find time tag
            time_tag = soup.find('time')
            date = time_tag.get('datetime', '') if time_tag else ''
        
        if not title or not text:
            return None
        
        return {
            'url': url,
            'title': title,
            'text': text,
            'date_publication': date,
            'portal': 'folha',
            'scraped_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ✗ Error scraping {url}: {e}")
        return None


def scrape_page(page_num: int, session: requests.Session) -> list:
    """
    Scrape all article links from a listing page
    
    Args:
        page_num: Page number
        session: Requests session
        
    Returns:
        List of article URLs
    """
    if page_num == 1:
        url = PAGE_1_URL
    else:
        url = BASE_URL.format(page_num)
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all article links (.shtml extension)
        links = soup.find_all('a', href=True)
        article_links = [
            a['href'] for a in links 
            if 'folha.uol.com.br' in a['href']
            and a['href'].endswith('.shtml')
            and '/institucional/' not in a['href']  # Exclude institutional pages
            and '/empreendedorsocial/' not in a['href']  # Exclude special sections
        ]
        
        # Remove duplicates
        article_links = list(set(article_links))
        
        return article_links
        
    except Exception as e:
        print(f"  ✗ Error on page {page_num}: {e}")
        return []


def find_max_page(session: requests.Session) -> int:
    """
    Find maximum number of pages using exponential search + binary search
    
    Args:
        session: Requests session
        
    Returns:
        Maximum page number with articles
    """
    print("\n" + "="*80)
    print("BUSCA AUTOMÁTICA DO NÚMERO MÁXIMO DE PÁGINAS")
    print("="*80 + "\n")
    
    # Phase 1: Exponential search to find upper bound
    print("Fase 1: Busca exponencial para encontrar limite superior...")
    test_page = 1
    
    while True:
        print(f"  Testando página {test_page}...", end=" ")
        links = scrape_page(test_page, session)
        
        if not links:
            print(f"✗ Sem artigos (parando)")
            break
        
        print(f"✓ {len(links)} artigos encontrados")
        test_page *= 2
        time.sleep(0.5)
    
    # Phase 2: Binary search to find exact max page
    left = test_page // 2
    right = test_page
    max_page = left
    
    print(f"\nFase 2: Busca binária no intervalo [{left}, {right}]...")
    
    while left <= right:
        mid = (left + right) // 2
        if mid == 0:
            break
        
        print(f"  Testando página {mid} (intervalo: {left}-{right})...", end=" ")
        links = scrape_page(mid, session)
        
        if links:
            print(f"✓ {len(links)} artigos encontrados")
            max_page = mid
            left = mid + 1
        else:
            print(f"✗ Sem artigos")
            right = mid - 1
        
        time.sleep(0.5)
    
    print(f"\n{'='*80}")
    print(f"✅ NÚMERO MÁXIMO DE PÁGINAS ENCONTRADO: {max_page}")
    print(f"{'='*80}\n")
    
    return max_page


def get_existing_urls() -> set:
    """Get URLs of already scraped articles"""
    existing_urls = set()
    
    if OUTPUT_DIR.exists():
        for json_file in OUTPUT_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'url' in data:
                        existing_urls.add(data['url'])
            except:
                continue
    
    return existing_urls


def save_article(article: dict, index: int):
    """Save article to JSON file"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = OUTPUT_DIR / f"folha_{index:04d}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(article, f, ensure_ascii=False, indent=2)


def main(max_pages: int = None, start_page: int = 1, auto_detect: bool = True):
    """
    Main scraping function
    
    Note: Folha de S.Paulo doesn't have real pagination - the homepage shows all recent articles.
    This scraper only scrapes the main page.
    
    Args:
        max_pages: Ignored (kept for compatibility)
        start_page: Ignored (kept for compatibility)
        auto_detect: Ignored (kept for compatibility)
    """
    print("\n" + "="*80)
    print("FOLHA DE S.PAULO NEWS SCRAPER")
    print("="*80 + "\n")
    
    # Setup
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    
    # Get existing articles
    existing_urls = get_existing_urls()
    print(f"Already scraped: {len(existing_urls)} articles\n")
    
    print("Note: Folha shows all articles on homepage (no pagination)\n")
    
    # Start scraping
    total_articles = len(existing_urls)
    new_articles = 0
    skipped = 0
    
    print("Scraping homepage:\n")
    
    # Get article links from homepage only
    article_urls = scrape_page(1, session)
    
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
    parser = argparse.ArgumentParser(description="Scrape Folha de S.Paulo news")
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help="Maximum number of pages to scrape (default: auto-detect)"
    )
    parser.add_argument(
        '--start-page',
        type=int,
        default=1,
        help="Starting page number (default: 1)"
    )
    parser.add_argument(
        '--no-auto-detect',
        action='store_true',
        help="Disable automatic page detection (use --max-pages value)"
    )
    
    args = parser.parse_args()
    
    main(
        max_pages=args.max_pages,
        start_page=args.start_page,
        auto_detect=not args.no_auto_detect
    )
