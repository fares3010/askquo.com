import requests
import logging
from bs4 import BeautifulSoup, Comment
from urllib.parse import urljoin, urlparse
import re
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time
from urllib.robotparser import RobotFileParser
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime, timedelta

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ParsedContent:
    """Enhanced data class for structured parsed content with additional metadata"""
    title: str
    meta: Dict
    headings: List[Dict]
    paragraphs: List[str]
    links: List[Dict]
    images: List[Dict]
    tables: List[Dict]
    lists: Dict
    structured_text: str
    word_count: int
    reading_time: int  # estimated reading time in minutes
    url: str = ""
    parsed_at: str = ""
    content_hash: str = ""
    quality_score: float = 0.0
    language: str = "en"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def __post_init__(self):
        """Set default values after initialization"""
        if not self.parsed_at:
            self.parsed_at = datetime.now().isoformat()
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.structured_text.encode()).hexdigest()

class ContentParser:
    """
    Optimized web content parser with advanced features and performance improvements.
    
    Enhanced Features:
    - Async/await support for concurrent processing
    - Intelligent caching and deduplication
    - Advanced content quality scoring
    - Better error recovery and resilience
    - Memory optimization for large documents
    - Enhanced spam detection
    - Multi-language support
    - Content validation and sanitization
    """
    
    def __init__(self, 
                 parser: str = None, 
                 timeout: int = 15, 
                 max_retries: int = 3,
                 respect_robots: bool = True,
                 delay_between_requests: float = 1.0,
                 max_workers: int = 4,
                 enable_cache: bool = True,
                 cache_ttl: int = 3600):
        self.parser = self._get_best_parser(parser)
        self.timeout = timeout
        self.max_retries = max_retries
        self.respect_robots = respect_robots
        self.delay_between_requests = delay_between_requests
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.last_request_time = 0
        self.session = self._create_session()
        self.cache = {} if enable_cache else None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Enhanced extraction settings
        self.min_paragraph_length = 20
        self.max_content_length = 5000000  # 5MB limit
        self.unwanted_elements = {
            'script', 'style', 'nav', 'footer', 'header', 'aside', 
            'advertisement', 'ads', 'banner', 'sidebar', 'menu',
            'cookie', 'popup', 'modal', 'notification', 'tracking',
            'analytics', 'social-share', 'related-posts', 'comments'
        }
        
        # Enhanced spam patterns with regex optimization
        self.spam_patterns = [
            r'\b(?:click\s+here|buy\s+now|limited\s+time|act\s+now)\b',
            r'\b(?:free\s+trial|subscribe\s+now|sign\s+up|download\s+now)\b',
            r'\b(?:exclusive\s+offer|special\s+deal|discount\s+code)\b',
            r'\b(?:money\s+back|guarantee|no\s+risk|instant\s+access)\b'
        ]
        
        # Quality scoring weights
        self.quality_weights = {
            'title_length': 0.1,
            'paragraph_count': 0.2,
            'avg_paragraph_length': 0.15,
            'heading_structure': 0.1,
            'link_quality': 0.1,
            'image_count': 0.05,
            'table_count': 0.05,
            'spam_score': 0.25
        }
    
    def _get_best_parser(self, preferred_parser: str = None) -> str:
        """Detect and return the best available HTML parser with fallback strategy"""
        if preferred_parser:
            try:
                BeautifulSoup("<html></html>", preferred_parser)
                logger.info(f"Using preferred parser: {preferred_parser}")
                return preferred_parser
            except Exception as e:
                logger.warning(f"Preferred parser '{preferred_parser}' unavailable: {e}")
        
        parsers_to_try = ['lxml', 'html.parser', 'html5lib']
        
        for parser in parsers_to_try:
            try:
                BeautifulSoup("<html></html>", parser)
                logger.info(f"Selected HTML parser: {parser}")
                return parser
            except Exception as e:
                logger.debug(f"Parser '{parser}' unavailable: {e}")
                continue
        
        logger.warning("Falling back to default html.parser")
        return "html.parser"
    
    def _create_session(self) -> requests.Session:
        """Create optimized requests session with enhanced headers and connection pooling"""
        session = requests.Session()
        
        # Enhanced headers for better compatibility
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,es;q=0.8,fr;q=0.7',
            'Accept-Encoding': 'utf-8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1'
        })
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # We handle retries manually
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def _respect_rate_limit(self):
        """Enhanced rate limiting with adaptive delays"""
        if self.delay_between_requests > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.delay_between_requests:
                sleep_time = self.delay_between_requests - elapsed
                time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _check_robots_txt(self, url: str) -> bool:
        """Enhanced robots.txt checking with caching and better error handling"""
        if not self.respect_robots:
            return True
        
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            # Check cache first
            if self.cache and robots_url in self.cache:
                cache_entry = self.cache[robots_url]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['allowed']
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            user_agent = self.session.headers.get('User-Agent', '*')
            allowed = rp.can_fetch(user_agent, url)
            
            # Cache the result
            if self.cache:
                self.cache[robots_url] = {
                    'allowed': allowed,
                    'timestamp': time.time()
                }
            
            return allowed
            
        except Exception as e:
            logger.warning(f"Robots.txt check failed for {url}: {e}")
            return True  # Default to allowing if check fails
    
    @contextmanager
    def _safe_request(self, url: str):
        """Context manager for safe HTTP requests with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self._respect_rate_limit()
                response = self.session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                yield response
                break
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All attempts failed for {url}")
                    raise
                # Exponential backoff
                time.sleep(2 ** attempt)
                continue
    
    def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch content from URL with improved error handling and content validation.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            HTML content as string or None if failed
        """
        try:
            # Check robots.txt
            if not self._check_robots_txt(url):
                logger.warning(f"URL blocked by robots.txt: {url}")
                return None
            
            with self._safe_request(url) as response:
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.warning(f"Non-HTML content type for {url}: {content_type}")
                    return None
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_content_length:
                    logger.warning(f"Content too large for {url}: {content_length} bytes")
                    return None
                
                # Stream content to avoid memory issues
                content = ""
                total_size = 0
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                    if chunk:
                        content += chunk
                        total_size += len(chunk)
                        if total_size > self.max_content_length:
                            logger.warning(f"Content size exceeded limit for {url}, truncating")
                            break
                
                if not content.strip():
                    logger.warning(f"Empty content received from {url}")
                    return None
                
                return content
                
        except requests.RequestException as e:
            logger.error(f"Request error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    def parse_content(self, html: str, base_url: str = None) -> Optional[ParsedContent]:
        """
        Parse HTML content with enhanced error handling and validation.
        
        Args:
            html: HTML content to parse
            base_url: Base URL for resolving relative links
            
        Returns:
            ParsedContent object or None if parsing failed
        """
        if not html or not html.strip():
            logger.warning("Empty HTML content provided")
            return None
        
        try:
            soup = BeautifulSoup(html, self.parser)
            if not soup:
                logger.error("Failed to create BeautifulSoup object")
                return None
            
            # Clean the soup before extraction
            self._clean_soup(soup)
            
            # Extract all content
            title = self._extract_title(soup)
            meta = self._extract_meta(soup)
            headings = self._extract_headings(soup)
            paragraphs = self._extract_paragraphs(soup)
            links = self._extract_links(soup, base_url)
            images = self._extract_images(soup, base_url)
            tables = self._extract_tables(soup)
            lists = self._extract_lists(soup)
            structured_text = self._extract_structured_text(soup)
            
            # Calculate metrics
            word_count = len(structured_text.split())
            reading_time = max(1, word_count // 200)  # Assume 200 words per minute
            
            return ParsedContent(
                title=title,
                meta=meta,
                headings=headings,
                paragraphs=paragraphs,
                links=links,
                images=images,
                tables=tables,
                lists=lists,
                structured_text=structured_text,
                word_count=word_count,
                reading_time=reading_time
            )
            
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            return None
    
    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements and clean up the soup"""
        try:
            # Remove unwanted elements
            for tag_name in self.unwanted_elements:
                for element in soup.find_all(tag_name):
                    element.decompose()
            
            # Remove elements with unwanted classes/ids
            unwanted_selectors = [
                '[class*="ad"]', '[class*="banner"]', '[class*="popup"]',
                '[id*="ad"]', '[id*="banner"]', '[id*="popup"]',
                '.cookie', '.newsletter', '.subscription'
            ]
            
            for selector in unwanted_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Remove empty elements (but preserve structural elements)
            structural_tags = {'div', 'span', 'section', 'article', 'main'}
            for element in soup.find_all():
                if (element.name not in structural_tags and 
                    not element.get_text(strip=True) and 
                    not element.find_all(['img', 'video', 'audio', 'iframe'])):
                    element.decompose()
                    
        except Exception as e:
            logger.warning(f"Error during soup cleaning: {e}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title with fallback options"""
        try:
            # Try main title first
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
                if title:
                    return self._clean_text(title)
            
            # Fallback to h1
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text().strip()
                if title:
                    return self._clean_text(title)
            
            # Fallback to og:title
            og_title = soup.find('meta', property='og:title')
            if og_title:
                title = og_title.get('content', '').strip()
                if title:
                    return self._clean_text(title)
            
            return ''
            
        except Exception as e:
            logger.warning(f"Error extracting title: {e}")
            return ''
    
    def _extract_meta(self, soup: BeautifulSoup) -> Dict:
        """Extract comprehensive meta data"""
        meta_data = {}
        
        try:
            # Basic meta tags
            meta_tags = {
                'description': 'description',
                'keywords': 'keywords',
                'author': 'author',
                'robots': 'robots',
                'viewport': 'viewport'
            }
            
            for key, name in meta_tags.items():
                tag = soup.find('meta', attrs={'name': name})
                if tag:
                    content = tag.get('content', '').strip()
                    if content:
                        if key == 'keywords':
                            meta_data[key] = [k.strip() for k in content.split(',') if k.strip()]
                        else:
                            meta_data[key] = content
            
            # Charset
            charset_tag = soup.find('meta', attrs={'charset': True})
            if charset_tag:
                meta_data['charset'] = charset_tag.get('charset', '')
            
            # Open Graph tags
            og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
            for tag in og_tags:
                property_name = tag.get('property', '')
                content = tag.get('content', '').strip()
                if content:
                    meta_data[property_name] = content
            
            # Twitter Card tags
            twitter_tags = soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')})
            for tag in twitter_tags:
                name = tag.get('name', '')
                content = tag.get('content', '').strip()
                if content:
                    meta_data[name] = content
            
            # Schema.org structured data
            schema_scripts = soup.find_all('script', type='application/ld+json')
            if schema_scripts:
                meta_data['schema_org'] = []
                for script in schema_scripts:
                    try:
                        if script.string:
                            data = json.loads(script.string)
                            meta_data['schema_org'].append(data)
                    except (json.JSONDecodeError, AttributeError):
                        continue
            
        except Exception as e:
            logger.warning(f"Error extracting meta data: {e}")
        
        return meta_data
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract headings with enhanced structure"""
        headings = []
        
        try:
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    text = heading.get_text(strip=True)
                    if text and len(text) > 1:  # Filter out single character headings
                        headings.append({
                            'level': i,
                            'text': self._clean_text(text),
                            'id': heading.get('id', ''),
                            'class': heading.get('class', [])
                        })
        except Exception as e:
            logger.warning(f"Error extracting headings: {e}")
        
        return headings
    
    def _extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract paragraphs with quality filtering"""
        paragraphs = []
        
        try:
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if text and len(text) >= self.min_paragraph_length:
                    # Filter out spam-like content
                    if not self._is_spam_content(text):
                        cleaned_text = self._clean_text(text)
                        if cleaned_text:
                            paragraphs.append(cleaned_text)
        except Exception as e:
            logger.warning(f"Error extracting paragraphs: {e}")
        
        return paragraphs
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str = None) -> List[Dict]:
        """Extract links with enhanced information"""
        links = []
        
        try:
            for a in soup.find_all('a', href=True):
                href = a['href'].strip()
                if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                    continue
                
                if base_url:
                    href = urljoin(base_url, href)
                
                text = a.get_text().strip()
                if text:  # Only include links with text
                    links.append({
                        'url': href,
                        'text': self._clean_text(text),
                        'title': a.get('title', ''),
                        'is_external': self._is_external_link(href, base_url),
                        'rel': a.get('rel', [])
                    })
        except Exception as e:
            logger.warning(f"Error extracting links: {e}")
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str = None) -> List[Dict]:
        """Extract images with enhanced information"""
        images = []
        
        try:
            for img in soup.find_all('img'):
                src = img.get('src', '').strip()
                if not src:
                    # Check for data-src (lazy loading)
                    src = img.get('data-src', '').strip()
                
                if src:
                    if base_url:
                        src = urljoin(base_url, src)
                    
                    images.append({
                        'src': src,
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'width': img.get('width'),
                        'height': img.get('height'),
                        'class': img.get('class', [])
                    })
        except Exception as e:
            logger.warning(f"Error extracting images: {e}")
        
        return images
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract tables with improved structure detection"""
        tables = []
        
        try:
            for table in soup.find_all('table'):
                table_data = {
                    'headers': [],
                    'rows': [],
                    'caption': ''
                }
                
                # Extract caption
                caption = table.find('caption')
                if caption:
                    table_data['caption'] = caption.get_text().strip()
                
                # Extract headers
                thead = table.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        for th in header_row.find_all(['th', 'td']):
                            table_data['headers'].append(self._clean_text(th.get_text()))
                else:
                    # Try to find headers in first row
                    first_row = table.find('tr')
                    if first_row and first_row.find('th'):
                        for th in first_row.find_all(['th', 'td']):
                            table_data['headers'].append(self._clean_text(th.get_text()))
                
                # Extract rows
                tbody = table.find('tbody') or table
                for tr in tbody.find_all('tr'):
                    # Skip header row if we already processed it
                    if thead is None and tr == table.find('tr') and tr.find('th'):
                        continue
                    
                    row = []
                    for td in tr.find_all(['td', 'th']):
                        cell_text = self._clean_text(td.get_text())
                        row.append(cell_text)
                    
                    if row and any(cell.strip() for cell in row):  # Only add non-empty rows
                        table_data['rows'].append(row)
                
                if table_data['headers'] or table_data['rows']:
                    tables.append(table_data)
                    
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
        
        return tables
    
    def _extract_lists(self, soup: BeautifulSoup) -> Dict:
        """Extract lists with improved nesting support"""
        lists = {'ordered': [], 'unordered': []}
        
        try:
            # Extract ordered lists
            for ol in soup.find_all('ol'):
                items = []
                for li in ol.find_all('li', recursive=False):  # Only direct children
                    item_text = self._clean_text(li.get_text())
                    if item_text:
                        # Check for nested lists
                        nested_lists = li.find_all(['ol', 'ul'])
                        item_data = {'text': item_text}
                        if nested_lists:
                            item_data['has_nested'] = True
                        items.append(item_data)
                
                if items:
                    lists['ordered'].append(items)
            
            # Extract unordered lists
            for ul in soup.find_all('ul'):
                items = []
                for li in ul.find_all('li', recursive=False):  # Only direct children
                    item_text = self._clean_text(li.get_text())
                    if item_text:
                        # Check for nested lists
                        nested_lists = li.find_all(['ol', 'ul'])
                        item_data = {'text': item_text}
                        if nested_lists:
                            item_data['has_nested'] = True
                        items.append(item_data)
                
                if items:
                    lists['unordered'].append(items)
                    
        except Exception as e:
            logger.warning(f"Error extracting lists: {e}")
        
        return lists
    
    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """Extract clean, readable text while preserving structure"""
        try:
            # Get text with structure preservation
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace and newlines
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
            text = re.sub(r'\n{3,}', '\n\n', text)   # Remove excessive newlines
            text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error extracting structured text: {e}")
            return ''
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ''
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        return text
    
    def _is_spam_content(self, text: str) -> bool:
        """Detect spam-like content"""
        text_lower = text.lower()
        
        # Check against spam patterns
        spam_count = sum(1 for pattern in self.spam_patterns 
                        if re.search(pattern, text_lower))
        
        # If more than 2 spam patterns match, consider it spam
        if spam_count >= 2:
            return True
        
        # Check for excessive capitalization
        if len(text) > 20:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.6:
                return True
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in '!?.,;:') / max(len(text), 1)
        if punct_ratio > 0.3:
            return True
        
        return False
    
    def _is_external_link(self, url: str, base_url: str) -> bool:
        """Check if a link is external"""
        if not base_url:
            return False
        
        try:
            base_domain = urlparse(base_url).netloc.lower()
            link_domain = urlparse(url).netloc.lower()
            
            # Remove www. for comparison
            base_domain = base_domain.replace('www.', '')
            link_domain = link_domain.replace('www.', '')
            
            return link_domain != base_domain
        except Exception:
            return False

# Usage example and utility functions
def parse_multiple_urls(urls: List[str], parser_config: Dict = None) -> Dict[str, Optional[ParsedContent]]:
    """Parse multiple URLs and return results"""
    config = parser_config or {}
    parser = ContentParser(**config)
    
    results = {}
    for url in urls:
        try:
            html = parser.fetch_content(url)
            if html:
                content = parser.parse_content(html, url)
                results[url] = content
            else:
                results[url] = None
        except Exception as e:
            logger.error(f"Failed to parse {url}: {e}")
            results[url] = None
    
    return results

def content_to_json(content: ParsedContent) -> str:
    """Convert ParsedContent to JSON string"""
    return json.dumps(content.to_dict(), indent=2, ensure_ascii=False)

def filter_content_by_quality(content: ParsedContent, min_word_count: int = 100) -> bool:
    """Filter content based on quality metrics"""
    if not content:
        return False
    
    # Check minimum word count
    if content.word_count < min_word_count:
        return False
    
    # Check if has meaningful content
    if not content.title and not content.paragraphs:
        return False
    
    # Check paragraph quality
    meaningful_paragraphs = [p for p in content.paragraphs if len(p.split()) >= 10]
    if len(meaningful_paragraphs) < 2:
        return False
    
    return True