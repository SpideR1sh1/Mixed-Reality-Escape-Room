"""
Massive Narrative & NLP Engine for Unity Integration
===================================================

This system handles massive-scale natural language processing and narrative generation:
- Dynamic story generation using large language models
- Real-time dialogue system with personality modeling
- Context-aware narrative adaptation
- Multi-layered storytelling (environmental, character, plot)
- Advanced sentiment analysis and player modeling
- Voice synthesis and audio generation
- Multi-language support with real-time translation
- Interactive narrative branching
- Procedural character generation with backstories
- Real-time plot adaptation based on player choices

Streams narrative content to Unity for immersive storytelling experiences.
"""

import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    pipeline
)
import asyncio
import websockets
import json
import threading
from queue import Queue
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import spacy
import networkx as nx
from collections import defaultdict
import sqlite3
import pickle
import re
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
import faiss
import gensim
from gensim.models import Word2Vec, Doc2Vec
import openai
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import librosa
import soundfile as sf

# Advanced narrative structures
from story_graph import StoryGraph, NarrativeNode, ChoiceNode
from character_system import CharacterPersonality, DialogueContext
from world_lore import LoreDatabase, EnvironmentalNarrative

@dataclass
class NarrativeContext:
    """Complete narrative context for story generation"""
    player_profile: Dict
    current_location: str
    recent_actions: List[str]
    emotional_state: str
    story_progress: float
    active_characters: List[str]
    environmental_factors: Dict
    available_objects: List[str]
    time_context: Dict
    player_relationships: Dict

class MassiveNarrativeEngine:
    """Main narrative generation and NLP processing engine"""
    
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # GPU/CPU device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🔥 Using device: {self.device}")
        
        # Large Language Models
        self.logger.info("📚 Loading large language models...")
        
        # Story generation model (GPT-Neo/GPT-J)
        self.story_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.story_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.story_model.to(self.device)
        
        # Dialogue generation model
        self.dialogue_tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")
        self.dialogue_model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-3B")
        self.dialogue_model.to(self.device)
        
        # Text summarization and adaptation
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
        
        # Sentiment analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=0 if torch.cuda.is_available() else -1)
        self.nltk_sentiment = SentimentIntensityAnalyzer()
        
        # Semantic similarity
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # NLP processing
        self.nlp = spacy.load("en_core_web_lg")
        
        # Specialized narrative systems
        self.story_architect = StoryArchitect(self.device)
        self.character_manager = CharacterManager(self.device)
        self.dialogue_engine = DialogueEngine(self.device)
        self.environmental_narrator = EnvironmentalNarrator(self.device)
        self.plot_adapter = PlotAdapter(self.device)
        self.voice_synthesizer = VoiceSynthesizer()
        
        # Narrative state management
        self.active_stories = {}
        self.character_memories = defaultdict(list)
        self.plot_state_database = sqlite3.connect('narrative_states.db', check_same_thread=False)
        
        # Real-time processing queues
        self.narrative_queue = Queue()
        self.dialogue_queue = Queue()
        self.voice_queue = Queue()
        
        # Performance monitoring
        self.generation_metrics = {
            'stories_generated': 0,
            'dialogue_exchanges': 0,
            'average_generation_time': 0,
            'active_narratives': 0
        }
        
        self._initialize_databases()
        self.logger.info("🚀 Massive Narrative Engine initialized successfully!")
    
    def _initialize_databases(self):
        """Initialize narrative databases and storage"""
        
        # Create narrative state tables
        cursor = self.plot_state_database.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS narrative_states (
                session_id TEXT PRIMARY KEY,
                story_progress REAL,
                character_relationships TEXT,
                plot_points_completed TEXT,
                emotional_history TEXT,
                player_choices TEXT,
                timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS character_memories (
                character_id TEXT,
                memory_content TEXT,
                emotional_weight REAL,
                timestamp REAL,
                context TEXT
            )
        ''')
        
        self.plot_state_database.commit()
    
    async def start_narrative_processing(self):
        """Start narrative generation and processing pipelines"""
        
        self.logger.info("📖 Starting narrative processing pipelines...")
        
        # Start processing threads
        narrative_thread = threading.Thread(
            target=self._narrative_generation_thread,
            daemon=True
        )
        narrative_thread.start()
        
        dialogue_thread = threading.Thread(
            target=self._dialogue_processing_thread,
            daemon=True
        )
        dialogue_thread.start()
        
        voice_thread = threading.Thread(
            target=self._voice_synthesis_thread,
            daemon=True
        )
        voice_thread.start()
        
        self.logger.info("✅ Narrative processing pipelines started!")
    
    async def generate_adaptive_narrative(self, context: NarrativeContext, session_id: str) -> Dict:
        """Generate adaptive narrative content based on context"""
        
        start_time = time.time()
        
        # Analyze player context for narrative direction
        narrative_direction = await self._analyze_narrative_direction(context)
        
        # Generate story content using multiple approaches
        story_tasks = [
            self.story_architect.generate_story_arc(context, narrative_direction),
            self.environmental_narrator.generate_environmental_story(context),
            self.character_manager.generate_character_interactions(context),
            self.plot_adapter.adapt_plot_to_player_actions(context, session_id)
        ]
        
        story_results = await asyncio.gather(*story_tasks)
        
        # Synthesize comprehensive narrative
        synthesized_narrative = await self._synthesize_narrative_elements(
            story_results, context, narrative_direction
        )
        
        # Generate contextual dialogue options
        dialogue_options = await self.dialogue_engine.generate_dialogue_options(
            context, synthesized_narrative
        )
        
        # Create environmental audio cues
        audio_cues = await self.voice_synthesizer.generate_narrative_audio(
            synthesized_narrative, context
        )
        
        # Update narrative state
        await self._update_narrative_state(session_id, context, synthesized_narrative)
        
        generation_time = time.time() - start_time
        self.generation_metrics['average_generation_time'] = (
            self.generation_metrics['average_generation_time'] * 0.9 + generation_time * 0.1
        )
        
        return {
            'narrative_content': synthesized_narrative,
            'dialogue_options': dialogue_options,
            'audio_cues': audio_cues,
            'narrative_direction': narrative_direction,
            'generation_time': generation_time,
            'adaptation_suggestions': await self._generate_adaptation_suggestions(context)
        }
    
    def _narrative_generation_thread(self):
        """Background thread for continuous narrative generation"""
        
        self.logger.info("📝 Narrative generation thread started")
        
        while True:
            try:
                if not self.narrative_queue.empty():
                    request = self.narrative_queue.get()
                    
                    # Generate narrative content
                    result = asyncio.run(self.generate_adaptive_narrative(
                        request['context'], request['session_id']
                    ))
                    
                    # Send result back to Unity
                    request['callback'](result)
                    
                    self.generation_metrics['stories_generated'] += 1
                
                time.sleep(0.01)  # Prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in narrative generation thread: {e}")
                time.sleep(0.1)

class StoryArchitect:
    """Advanced story architecture and plot generation"""
    
    def __init__(self, device):
        self.device = device
        
        # Story structure templates
        self.story_structures = self._load_story_structures()
        self.plot_templates = self._load_plot_templates()
        self.narrative_patterns = self._load_narrative_patterns()
        
        # Genre-specific generators
        self.genre_generators = {
            'mystery': MysteryGenerator(device),
            'adventure': AdventureGenerator(device),
            'horror': HorrorGenerator(device),
            'drama': DramaGenerator(device),
            'comedy': ComedyGenerator(device),
            'scifi': SciFiGenerator(device)
        }
    
    async def generate_story_arc(self, context: NarrativeContext, direction: Dict) -> Dict:
        """Generate comprehensive story arc with multiple plot threads"""
        
        # Determine optimal story structure
        story_structure = await self._select_optimal_structure(context, direction)
        
        # Generate main plot thread
        main_plot = await self._generate_main_plot_thread(context, story_structure)
        
        # Generate subplot threads
        subplots = await self._generate_subplot_threads(context, main_plot)
        
        # Create character arcs
        character_arcs = await self._generate_character_arcs(context, main_plot, subplots)
        
        # Generate plot points and beats
        plot_points = await self._generate_plot_points(main_plot, subplots, character_arcs)
        
        # Create narrative pacing
        pacing_structure = await self._create_pacing_structure(plot_points, context)
        
        return {
            'story_structure': story_structure,
            'main_plot': main_plot,
            'subplots': subplots,
            'character_arcs': character_arcs,
            'plot_points': plot_points,
            'pacing_structure': pacing_structure,
            'estimated_duration': self._calculate_story_duration(plot_points, pacing_structure)
        }
    
    async def _generate_main_plot_thread(self, context: NarrativeContext, structure: Dict) -> Dict:
        """Generate the main plot thread using advanced story generation"""
        
        # Create plot prompt based on context
        plot_prompt = self._create_plot_prompt(context, structure)
        
        # Generate story using language model
        story_segments = []
        
        for beat in structure['story_beats']:
            beat_prompt = f"{plot_prompt}\n\nStory beat: {beat['description']}\nGenerate narrative:"
            
            # Generate story segment
            story_segment = await self._generate_story_segment(beat_prompt, beat['target_length'])
            
            # Analyze and enhance segment
            enhanced_segment = await self._enhance_story_segment(story_segment, context, beat)
            
            story_segments.append({
                'beat_type': beat['type'],
                'content': enhanced_segment,
                'emotional_target': beat['emotional_target'],
                'plot_advancement': beat['plot_advancement']
            })
        
        return {
            'theme': structure['theme'],
            'genre': structure['genre'],
            'segments': story_segments,
            'central_conflict': await self._identify_central_conflict(story_segments),
            'resolution_path': await self._plan_resolution_path(story_segments)
        }

class CharacterManager:
    """Advanced character personality and interaction management"""
    
    def __init__(self, device):
        self.device = device
        
        # Character personality models
        self.personality_analyzer = PersonalityAnalyzer()
        self.relationship_tracker = RelationshipTracker()
        self.character_database = CharacterDatabase()
        
        # Dialogue style models
        self.dialogue_styles = self._load_dialogue_styles()
        self.speech_patterns = self._load_speech_patterns()
    
    async def generate_character_interactions(self, context: NarrativeContext) -> Dict:
        """Generate character interactions based on context and relationships"""
        
        active_characters = context.active_characters
        
        if not active_characters:
            return {'has_interactions': False}
        
        interactions = []
        
        # Generate interactions between characters
        for i, char1 in enumerate(active_characters):
            for char2 in active_characters[i+1:]:
                interaction = await self._generate_character_interaction(
                    char1, char2, context
                )
                if interaction['significance'] > 0.3:  # Only include meaningful interactions
                    interactions.append(interaction)
        
        # Generate player-character interactions
        player_interactions = []
        for character in active_characters:
            interaction = await self._generate_player_character_interaction(
                character, context
            )
            if interaction['relevance'] > 0.4:
                player_interactions.append(interaction)
        
        return {
            'has_interactions': len(interactions) > 0 or len(player_interactions) > 0,
            'character_interactions': interactions,
            'player_interactions': player_interactions,
            'relationship_changes': await self._calculate_relationship_changes(interactions, context),
            'character_developments': await self._analyze_character_development(interactions, context)
        }
    
    async def _generate_character_interaction(self, char1: str, char2: str, context: NarrativeContext) -> Dict:
        """Generate interaction between two characters"""
        
        # Get character personalities
        char1_personality = await self.character_database.get_character_personality(char1)
        char2_personality = await self.character_database.get_character_personality(char2)
        
        # Get relationship history
        relationship = await self.relationship_tracker.get_relationship(char1, char2)
        
        # Generate interaction based on personalities and context
        interaction_prompt = self._create_interaction_prompt(
            char1_personality, char2_personality, relationship, context
        )
        
        # Generate dialogue
        dialogue = await self._generate_character_dialogue(
            interaction_prompt, char1_personality, char2_personality
        )
        
        # Analyze interaction significance
        significance = await self._analyze_interaction_significance(
            dialogue, char1_personality, char2_personality, context
        )
        
        return {
            'characters': [char1, char2],
            'dialogue': dialogue,
            'significance': significance,
            'emotional_impact': await self._calculate_emotional_impact(dialogue),
            'relationship_effect': await self._predict_relationship_effect(dialogue, relationship)
        }

class DialogueEngine:
    """Advanced dialogue generation and conversation management"""
    
    def __init__(self, device):
        self.device = device
        
        # Conversation context tracking
        self.conversation_contexts = {}
        self.dialogue_history = defaultdict(list)
        
        # Personality-based dialogue generation
        self.personality_dialogue_models = {}
        
        # Emotion and sentiment tracking
        self.emotion_tracker = EmotionTracker()
        self.conversation_flow_analyzer = ConversationFlowAnalyzer()
    
    async def generate_dialogue_options(self, context: NarrativeContext, narrative: Dict) -> Dict:
        """Generate contextual dialogue options for player"""
        
        # Analyze current conversation state
        conversation_state = await self._analyze_conversation_state(context)
        
        # Generate multiple dialogue options with different approaches
        dialogue_options = []
        
        # Generate options based on different conversation strategies
        strategies = ['direct', 'diplomatic', 'aggressive', 'curious', 'empathetic']
        
        for strategy in strategies:
            option = await self._generate_dialogue_option(
                context, narrative, conversation_state, strategy
            )
            
            if option['quality_score'] > 0.5:
                dialogue_options.append(option)
        
        # Generate context-specific options
        contextual_options = await self._generate_contextual_options(
            context, narrative, conversation_state
        )
        
        dialogue_options.extend(contextual_options)
        
        # Rank and filter options
        ranked_options = await self._rank_dialogue_options(
            dialogue_options, context, conversation_state
        )
        
        return {
            'dialogue_options': ranked_options[:6],  # Top 6 options
            'conversation_state': conversation_state,
            'predicted_responses': await self._predict_npc_responses(ranked_options, context),
            'relationship_implications': await self._analyze_relationship_implications(ranked_options, context)
        }
    
    async def _generate_dialogue_option(self, context: NarrativeContext, narrative: Dict, 
                                      conversation_state: Dict, strategy: str) -> Dict:
        """Generate a single dialogue option using specified strategy"""
        
        # Create strategy-specific prompt
        strategy_prompt = self._create_strategy_prompt(
            context, narrative, conversation_state, strategy
        )
        
        # Generate dialogue using language model
        generated_dialogue = await self._generate_strategic_dialogue(
            strategy_prompt, strategy
        )
        
        # Analyze dialogue quality and implications
        quality_analysis = await self._analyze_dialogue_quality(
            generated_dialogue, strategy, context
        )
        
        return {
            'text': generated_dialogue,
            'strategy': strategy,
            'quality_score': quality_analysis['quality_score'],
            'emotional_tone': quality_analysis['emotional_tone'],
            'predicted_reaction': quality_analysis['predicted_reaction'],
            'relationship_impact': quality_analysis['relationship_impact']
        }

class VoiceSynthesizer:
    """Advanced voice synthesis and audio generation"""
    
    def __init__(self):
        # Text-to-speech engines
        self.tts_engine = pyttsx3.init()
        self.voices = self.tts_engine.getProperty('voices')
        
        # Character-specific voice mapping
        self.character_voices = {}
        
        # Audio processing
        self.audio_processor = AudioProcessor()
        
        # Emotional speech modification
        self.emotion_modifier = EmotionalSpeechModifier()
    
    async def generate_narrative_audio(self, narrative: Dict, context: NarrativeContext) -> Dict:
        """Generate audio cues and narration for narrative content"""
        
        audio_cues = []
        
        # Generate environmental narration
        if narrative.get('environmental_story'):
            env_audio = await self._generate_environmental_narration(
                narrative['environmental_story'], context
            )
            audio_cues.extend(env_audio)
        
        # Generate character dialogue audio
        if narrative.get('character_interactions'):
            dialogue_audio = await self._generate_dialogue_audio(
                narrative['character_interactions'], context
            )
            audio_cues.extend(dialogue_audio)
        
        # Generate atmospheric audio suggestions
        atmospheric_audio = await self._generate_atmospheric_audio(
            narrative, context
        )
        audio_cues.extend(atmospheric_audio)
        
        return {
            'audio_cues': audio_cues,
            'total_duration': sum(cue['duration'] for cue in audio_cues),
            'audio_timeline': await self._create_audio_timeline(audio_cues),
            'spatial_audio_data': await self._generate_spatial_audio_data(audio_cues, context)
        }

# Unity WebSocket Integration
class UnityNarrativeStreamer:
    """Stream narrative content to Unity for real-time storytelling"""
    
    def __init__(self, narrative_engine):
        self.narrative_engine = narrative_engine
        self.connected_clients = set()
        self.active_sessions = {}
    
    async def start_server(self):
        """Start WebSocket server for Unity narrative integration"""
        
        print("🚀 Starting Massive Narrative Engine Server on ws://localhost:8891")
        
        async def handle_unity_client(websocket, path):
            self.connected_clients.add(websocket)
            print(f"🎮 Unity connected for narrative streaming. Active clients: {len(self.connected_clients)}")
            
            try:
                # Send initial narrative capabilities
                capabilities = {
                    'type': 'narrative_capabilities',
                    'features': [
                        'dynamic_story_generation',
                        'adaptive_dialogue',
                        'character_interactions',
                        'environmental_storytelling',
                        'voice_synthesis',
                        'emotional_analysis',
                        'plot_adaptation'
                    ]
                }
                await websocket.send(json.dumps(capabilities))
                
                # Handle Unity narrative requests
                async for message in websocket:
                    unity_request = json.loads(message)
                    await self.handle_unity_narrative_request(unity_request, websocket)
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.connected_clients.remove(websocket)
                print(f"🎮 Unity disconnected from narrative engine. Active clients: {len(self.connected_clients)}")
        
        # Start WebSocket server
        server = await websockets.serve(handle_unity_client, "localhost", 8891)
        print("📖 Narrative Engine ready for Unity connection!")
        
        await server.wait_closed()
    
    async def handle_unity_narrative_request(self, request, websocket):
        """Handle narrative requests from Unity"""
        
        request_type = request.get('type')
        
        if request_type == 'generate_narrative':
            # Unity requesting narrative generation
            context = NarrativeContext(**request['context'])
            session_id = request.get('session_id', 'default')
            
            narrative_result = await self.narrative_engine.generate_adaptive_narrative(
                context, session_id
            )
            
            response = {
                'type': 'narrative_result',
                'session_id': session_id,
                'data': narrative_result
            }
            
            await websocket.send(json.dumps(response))
        
        elif request_type == 'process_player_choice':
            # Unity sending player dialogue choice
            choice_data = request['choice_data']
            session_id = request['session_id']
            
            # Process choice and generate response
            choice_result = await self.narrative_engine.process_player_choice(
                choice_data, session_id
            )
            
            response = {
                'type': 'choice_result',
                'session_id': session_id,
                'data': choice_result
            }
            
            await websocket.send(json.dumps(response))
        
        elif request_type == 'request_dialogue_options':
            # Unity requesting dialogue options for current context
            context = NarrativeContext(**request['context'])
            session_id = request['session_id']
            
            dialogue_options = await self.narrative_engine.dialogue_engine.generate_dialogue_options(
                context, {}
            )
            
            response = {
                'type': 'dialogue_options',
                'session_id': session_id,
                'data': dialogue_options
            }
            
            await websocket.send(json.dumps(response))

# Unity Integration Script
unity_narrative_script = '''
/*
 * MassiveNarrativeClient.cs
 * Unity client for massive Python narrative engine
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;
using System;
using UnityEngine.Events;

public class MassiveNarrativeClient : MonoBehaviour
{
    private WebSocket narrativeSocket;
    private string currentSessionId;
    
    [Header("Narrative Settings")]
    public string pythonNarrativeUrl = "ws://localhost:8891";
    public bool autoConnect = true;
    public bool enableVoiceSynthesis = true;
    public bool enableDynamicStory = true;
    
    [Header("Player Context")]
    public Transform playerTransform;
    public string currentLocation = "cave_entrance";
    public List<string> recentActions = new List<string>();
    
    [Header("UI References")]
    public Transform dialogueUI;
    public AudioSource narrativeAudioSource;
    
    // Events
    public UnityEvent<NarrativeResult> OnNarrativeGenerated;
    public UnityEvent<List<DialogueOption>> OnDialogueOptionsReceived;
    public UnityEvent<string> OnStoryUpdate;
    
    [Serializable]
    public class NarrativeResult
    {
        public NarrativeContent narrative_content;
        public List<DialogueOption> dialogue_options;
        public List<AudioCue> audio_cues;
        public float generation_time;
    }
    
    [Serializable]
    public class DialogueOption
    {
        public string text;
        public string strategy;
        public float quality_score;
        public string emotional_tone;
        public float relationship_impact;
    }
    
    void Start()
    {
        currentSessionId = $"session_{System.DateTime.Now.Ticks}";
        
        if (autoConnect)
        {
            ConnectToNarrativeEngine();
        }
    }
    
    public void ConnectToNarrativeEngine()
    {
        Debug.Log("📖 Connecting to Massive Narrative Engine...");
        
        narrativeSocket = new WebSocket(pythonNarrativeUrl);
        
        narrativeSocket.OnOpen += (sender, e) =>
        {
            Debug.Log("✅ Connected to Narrative Engine!");
            RequestInitialNarrative();
        };
        
        narrativeSocket.OnMessage += (sender, e) =>
        {
            try
            {
                var message = JsonConvert.DeserializeObject<Dictionary<string, object>>(e.Data);
                string messageType = message["type"].ToString();
                
                switch (messageType)
                {
                    case "narrative_capabilities":
                        ProcessNarrativeCapabilities(message);
                        break;
                    case "narrative_result":
                        ProcessNarrativeResult(message);
                        break;
                    case "dialogue_options":
                        ProcessDialogueOptions(message);
                        break;
                    case "choice_result":
                        ProcessChoiceResult(message);
                        break;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"Error processing narrative data: {ex.Message}");
            }
        };
        
        narrativeSocket.Connect();
    }
    
    private void RequestInitialNarrative()
    {
        var context = CreateCurrentContext();
        
        var request = new Dictionary<string, object>
        {
            ["type"] = "generate_narrative",
            ["session_id"] = currentSessionId,
            ["context"] = context
        };
        
        narrativeSocket.Send(JsonConvert.SerializeObject(request));
    }
    
    private Dictionary<string, object> CreateCurrentContext()
    {
        return new Dictionary<string, object>
        {
            ["player_profile"] = GetPlayerProfile(),
            ["current_location"] = currentLocation,
            ["recent_actions"] = recentActions.ToArray(),
            ["emotional_state"] = GetPlayerEmotionalState(),
            ["story_progress"] = GetStoryProgress(),
            ["active_characters"] = GetActiveCharacters(),
            ["environmental_factors"] = GetEnvironmentalFactors(),
            ["available_objects"] = GetAvailableObjects(),
            ["time_context"] = GetTimeContext(),
            ["player_relationships"] = GetPlayerRelationships()
        };
    }
    
    public void SelectDialogueOption(int optionIndex)
    {
        if (currentDialogueOptions != null && optionIndex < currentDialogueOptions.Count)
        {
            var selectedOption = currentDialogueOptions[optionIndex];
            
            var choiceData = new Dictionary<string, object>
            {
                ["selected_option"] = selectedOption,
                ["option_index"] = optionIndex,
                ["timestamp"] = Time.time,
                ["context"] = CreateCurrentContext()
            };
            
            var request = new Dictionary<string, object>
            {
                ["type"] = "process_player_choice",
                ["session_id"] = currentSessionId,
                ["choice_data"] = choiceData
            };
            
            narrativeSocket.Send(JsonConvert.SerializeObject(request));
            
            // Add to recent actions
            recentActions.Add($"dialogue_choice:{selectedOption.text}");
            if (recentActions.Count > 10) recentActions.RemoveAt(0);
        }
    }
    
    private void ProcessNarrativeResult(Dictionary<string, object> message)
    {
        var narrativeResult = JsonConvert.DeserializeObject<NarrativeResult>(
            message["data"].ToString()
        );
        
        // Trigger narrative events
        OnNarrativeGenerated?.Invoke(narrativeResult);
        
        // Play audio cues
        if (enableVoiceSynthesis && narrativeResult.audio_cues != null)
        {
            StartCoroutine(PlayAudioCues(narrativeResult.audio_cues));
        }
        
        // Update story UI
        if (narrativeResult.narrative_content != null)
        {
            UpdateStoryDisplay(narrativeResult.narrative_content);
        }
        
        // Cache dialogue options
        currentDialogueOptions = narrativeResult.dialogue_options;
        OnDialogueOptionsReceived?.Invoke(currentDialogueOptions);
    }
    
    void Update()
    {
        // Continuously update context and request new narrative content
        if (Time.time % 10.0f < Time.deltaTime) // Every 10 seconds
        {
            if (HasContextChanged())
            {
                RequestContextualNarrative();
            }
        }
    }
    
    public void TriggerStoryEvent(string eventType, Dictionary<string, object> eventData)
    {
        recentActions.Add($"story_event:{eventType}");
        RequestContextualNarrative();
    }
    
    private void RequestContextualNarrative()
    {
        var request = new Dictionary<string, object>
        {
            ["type"] = "generate_narrative",
            ["session_id"] = currentSessionId,
            ["context"] = CreateCurrentContext()
        };
        
        narrativeSocket.Send(JsonConvert.SerializeObject(request));
    }
}
'''

async def main():
    """Start the massive narrative engine"""
    
    print("📖 MASSIVE NARRATIVE & NLP ENGINE FOR UNITY")
    print("==========================================")
    print("This system provides:")
    print("• Dynamic story generation (GPT-Neo/Blenderbot)")
    print("• Adaptive dialogue system")
    print("• Character personality modeling")
    print("• Environmental storytelling")
    print("• Voice synthesis and audio generation")
    print("• Real-time plot adaptation")
    print("• Multi-language support")
    print("• Sentiment and emotion analysis")
    
    # Initialize narrative engine
    narrative_engine = MassiveNarrativeEngine()
    
    # Start narrative processing
    await narrative_engine.start_narrative_processing()
    
    # Initialize Unity streamer
    unity_streamer = UnityNarrativeStreamer(narrative_engine)
    
    # Start server
    await unity_streamer.start_server()

if __name__ == "__main__":
    asyncio.run(main()) 