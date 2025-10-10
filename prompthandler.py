"""
Enhanced prompt handler for Aurora with better intent detection
"""
import re
import logging
from typing import Optional, Dict, Any
from PreTrainedResponses import get_pretrained_response
from Generation import ollama_manager
from ws import web_services
from logmanagement import log_manager

logger = logging.getLogger(__name__)

class PromptHandler:
    """Enhanced prompt handler with better intent detection and routing"""
    
    def __init__(self):
        self.intent_patterns = {
            'wikipedia': [
                r'^wikipedia\s+(.+)',
                r'^wiki\s+(.+)',
                r'^search wikipedia for\s+(.+)',
                r'^look up\s+(.+)\s+on wikipedia'
            ],
            'google_search': [
                r'^search google for\s+(.+)',
                r'^google\s+(.+)',
                r'^search\s+(.+)\s+on google'
            ],
            'youtube_search': [
                r'^search youtube for\s+(.+)',
                r'^youtube\s+(.+)',
                r'^play\s+(.+)\s+on youtube',
                r'^find\s+(.+)\s+on youtube'
            ],
            'web_search': [
                r'^web search\s+(.+)',
                r'^search web for\s+(.+)',
                r'^search for\s+(.+)',
                r'^look up\s+(.+)',
                r'^find information about\s+(.+)'
            ],
            'smart_search': [
                r'^smart search\s+(.+)',
                r'^deep search\s+(.+)',
                r'^research\s+(.+)'
            ],
            'news': [
                r'^news$',
                r'^latest news$',
                r'^get news$',
                r'^what\'?s in the news',
                r'^news from\s+(.+)',
                r'^(.+)\s+news$'
            ],
            'news_topic': [
                r'^news about\s+(.+)',
                r'^search news for\s+(.+)',
                r'^(.+)\s+in the news$'
            ],
            'weather': [
                r'^weather$',
                r'^weather in\s+(.+)',
                r'^what\'?s the weather in\s+(.+)',
                r'^what is the weather in\s+(.+)',
                r'^what\'?s the weather like in\s+(.+)',
                r'^what is the weather like in\s+(.+)',
                r'^(.+)\s+weather$',
                r'^how\'?s the weather in\s+(.+)',
                r'^how is the weather in\s+(.+)',
                r'^tell me about the weather in\s+(.+)',
                r'^get weather for\s+(.+)'
            ]
        }
    
    def _extract_intent_and_params(self, query: str) -> tuple[Optional[str], Optional[str]]:
        """Extract intent and parameters from query using regex patterns"""
        query_lower = query.lower().strip()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.match(pattern, query_lower)
                if match:
                    # Extract the captured group (parameter) if available
                    param = match.group(1) if match.groups() else None
                    return intent, param
        
        return None, None
    
    def _handle_wikipedia_search(self, topic: str, original_query: str) -> str:
        """Handle Wikipedia search requests"""
        if not topic:
            # Try to extract topic from the original query
            topic = original_query.lower().replace('wikipedia', '').strip()
        
        if not topic:
            return "Please specify what you'd like me to search for on Wikipedia."
        
        logger.info(f"Wikipedia search for: {topic}")
        result = web_services.search_wikipedia(topic)
        
        if result:
            return f"📖 **Wikipedia Summary for '{topic}':**\n\n{result}"
        else:
            return f"Sorry, I couldn't find information about '{topic}' on Wikipedia."
    
    def _handle_google_search(self, term: str, original_query: str) -> str:
        """Handle Google search requests"""
        if not term:
            term = original_query.lower().replace('search google for', '').replace('google', '').strip()
        
        if not term:
            return "Please specify what you'd like me to search for on Google."
        
        logger.info(f"Google search for: {term}")
        return web_services.open_google(term)
    
    def _handle_youtube_search(self, term: str, original_query: str) -> str:
        """Handle YouTube search requests"""
        if not term:
            term = original_query.lower().replace('search youtube for', '').replace('youtube', '').strip()
        
        if not term:
            return "Please specify what you'd like me to search for on YouTube."
        
        logger.info(f"YouTube search for: {term}")
        return web_services.open_youtube(term)
    
    def _handle_web_search(self, query: str, original_query: str) -> str:
        """Handle AI-powered web search requests"""
        if not query:
            query = original_query
        
        if not query:
            return "Please specify what you'd like me to search for."
        
        logger.info(f"AI web search for: {query}")
        return web_services.ai_web_search(query)
    
    def _handle_smart_search(self, query: str, original_query: str) -> str:
        """Handle smart search requests (Wikipedia + AI)"""
        if not query:
            query = original_query
        
        if not query:
            return "Please specify what you'd like me to research."
        
        logger.info(f"Smart search for: {query}")
        return web_services.smart_search(query)
    
    def _handle_news_topic(self, topic: str) -> str:
        """Handle news search by topic using AI-powered web search"""
        if not topic:
            return "Please specify a news topic to search for."
        
        logger.info(f"News topic search for: {topic}")
        # get_news_by_topic now returns a formatted string, not a list
        news_text = web_services.get_news_by_topic(topic)
        
        if news_text:
            return news_text
        else:
            return f"Sorry, I couldn't find recent news about '{topic}'."
    
    def _handle_news_request(self, location: Optional[str] = None) -> str:
        """Handle news requests using AI-powered web search"""
        logger.info(f"News request for location: {location}")
        
        country = 'us'  # default
        if location:
            # Simple country mapping
            country_map = {
                'india': 'in',
                'uk': 'gb',
                'britain': 'gb',
                'canada': 'ca',
                'australia': 'au',
                'germany': 'de',
                'france': 'fr',
                'japan': 'jp',
                'china': 'cn'
            }
            country = country_map.get(location.lower(), 'us')
        
        # get_news now returns a formatted string, not a list
        news_text = web_services.get_news(country=country)
        
        if news_text:
            return news_text
        else:
            return "Sorry, I couldn't fetch the latest news right now."
    
    def _handle_weather_request(self, city: Optional[str]) -> str:
        """Handle weather requests using AI-powered web search"""
        if not city:
            return "Please specify a city for the weather forecast. For example: 'weather in New York'"
        
        logger.info(f"Weather request for: {city}")
        # get_weather now returns a formatted string directly
        weather_info = web_services.get_weather(city)
        
        if weather_info:
            return weather_info
        else:
            return f"Sorry, I couldn't get weather information for '{city}'. Please check the city name."
    
    def handle_query(self, query: str) -> str:
        """
        Main query handler with enhanced intent detection
        
        Args:
            query: User's input query
            
        Returns:
            Response string
        """
        if not query or not query.strip():
            return "Please ask me something!"
        
        query = query.strip()
        logger.info(f"Processing query: {query[:100]}{'...' if len(query) > 100 else ''}")
        log_manager.append_md_log("Query Processing", f"Intent detection for: {query}")
        
        # Check for pre-trained responses first
        pretrained_response = get_pretrained_response(query)
        if pretrained_response:
            logger.info("Using pre-trained response")
            log_manager.append_md_log("Response Type", "Pre-trained")
            return pretrained_response
        
        # Extract intent and parameters
        intent, param = self._extract_intent_and_params(query)
        
        if intent:
            logger.info(f"Detected intent: {intent} with param: {param}")
            log_manager.append_md_log("Intent Detected", f"{intent} - {param}")
            
            # Route to appropriate handler
            if intent == 'wikipedia':
                return self._handle_wikipedia_search(param, query)
            
            elif intent == 'google_search':
                return self._handle_google_search(param, query)
            
            elif intent == 'youtube_search':
                return self._handle_youtube_search(param, query)
            
            elif intent == 'web_search':
                return self._handle_web_search(param, query)
            
            elif intent == 'smart_search':
                return self._handle_smart_search(param, query)
            
            elif intent == 'news':
                return self._handle_news_request(param)
            
            elif intent == 'news_topic':
                return self._handle_news_topic(param)
            
            elif intent == 'weather':
                return self._handle_weather_request(param)
        
        # Fallback to LLM for complex queries
        logger.info("No specific intent detected, using LLM")
        log_manager.append_md_log("Response Type", "LLM Generation")
        
        try:
            response = ollama_manager.chat_with_memory(query)
            return response
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return "Sorry, I encountered an error while processing your request. Please try again."
    
    def get_help_text(self) -> str:
        """Get help text with available commands"""
        help_text = """
🌅 **Aurora Help**

**Available Commands:**

🔍 **Search Commands:**
- `wikipedia [topic]` - Search Wikipedia
- `search google for [term]` - Open Google search
- `search youtube for [term]` - Open YouTube search
- `web search [query]` - AI-powered web search (NEW!)
- `smart search [query]` - Combined Wikipedia + AI search (NEW!)
- `search for [anything]` - General web search

📰 **Information Commands:**
- `news` - Get latest news headlines
- `news about [topic]` - Search news by topic (NEW!)
- `weather in [city]` - Get weather information

💬 **Chat:**
- Just type anything else to chat with the AI!

**Examples:**
- "wikipedia artificial intelligence"
- "web search latest AI developments"
- "smart search quantum computing"
- "news about climate change"
- "weather in London"
- "search youtube for funny cats"
- "What is machine learning?"
"""
        return help_text

# Create global instance
prompt_handler = PromptHandler()

# Backward compatibility function
def handle_query(query: str) -> str:
    """Backward compatibility function"""
    return prompt_handler.handle_query(query)

