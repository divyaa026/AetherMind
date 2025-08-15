import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Mic, Send, Heart, Brain, Zap } from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import AetherMindAPIService, { EmotionEntry } from '@/services/AetherMindAPI';

interface EmotionData {
  name: string;
  color: string;
  intensity: number;
  category: 'joy' | 'sadness' | 'fear' | 'anger' | 'surprise' | 'trust';
}

const EmotionWheel: React.FC = () => {
  const [selectedEmotion, setSelectedEmotion] = useState<EmotionData | null>(null);
  const [journalText, setJournalText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [sentiment, setSentiment] = useState<'positive' | 'neutral' | 'negative'>('neutral');

  const emotions: EmotionData[] = [
    // Joy family
    { name: 'joyful', color: 'bg-yellow-400', intensity: 0.9, category: 'joy' },
    { name: 'hopeful', color: 'bg-yellow-300', intensity: 0.7, category: 'joy' },
    { name: 'content', color: 'bg-yellow-200', intensity: 0.5, category: 'joy' },
    
    // Sadness family  
    { name: 'melancholy', color: 'bg-blue-400', intensity: 0.6, category: 'sadness' },
    { name: 'lonely', color: 'bg-blue-500', intensity: 0.7, category: 'sadness' },
    { name: 'despair', color: 'bg-blue-600', intensity: 0.9, category: 'sadness' },
    
    // Fear family
    { name: 'anxious', color: 'bg-purple-400', intensity: 0.7, category: 'fear' },
    { name: 'worried', color: 'bg-purple-300', intensity: 0.5, category: 'fear' },
    { name: 'terrified', color: 'bg-purple-600', intensity: 0.9, category: 'fear' },
    
    // Anger family
    { name: 'frustrated', color: 'bg-red-400', intensity: 0.6, category: 'anger' },
    { name: 'irritated', color: 'bg-red-300', intensity: 0.4, category: 'anger' },
    { name: 'furious', color: 'bg-red-600', intensity: 0.9, category: 'anger' },
    
    // Trust family
    { name: 'calm', color: 'bg-green-300', intensity: 0.5, category: 'trust' },
    { name: 'confident', color: 'bg-green-400', intensity: 0.7, category: 'trust' },
    { name: 'peaceful', color: 'bg-green-200', intensity: 0.6, category: 'trust' },
  ];

  const handleEmotionSelect = (emotion: EmotionData) => {
    setSelectedEmotion(emotion);
    // Auto-determine sentiment based on emotion category
    const newSentiment = ['joy', 'trust'].includes(emotion.category) ? 'positive' : 
                        emotion.category === 'sadness' ? 'negative' : 'neutral';
    setSentiment(newSentiment);
  };

  const handleTextAnalysis = async (text: string) => {
    if (!text.trim()) return;
    
    setIsAnalyzing(true);
    try {
      const analysis = await AetherMindAPIService.analyzeText(text);
      
      // Update sentiment based on analysis
      if (analysis.riskLevel > 0.6) {
        setSentiment('negative');
      } else if (analysis.flags.includes('positive_sentiment')) {
        setSentiment('positive');
      }
      
      // Show recommendations if high risk
      if (analysis.riskLevel > 0.7 && analysis.recommendedActions) {
        toast({
          title: "We're here for you",
          description: analysis.recommendedActions[0],
          duration: 8000,
        });
      }
    } catch (error) {
      console.error('Analysis failed:', error);
    }
    setIsAnalyzing(false);
  };

  const handleVoiceInput = () => {
    setIsListening(true);
    // Simulate voice input
    setTimeout(() => {
      setJournalText("I've been feeling really overwhelmed lately");
      setIsListening(false);
      handleTextAnalysis("I've been feeling really overwhelmed lately");
    }, 2000);
  };

  const handleSaveEntry = async () => {
    if (!selectedEmotion || !journalText.trim()) {
      toast({
        title: "Missing information",
        description: "Please select an emotion and write about your feelings",
        variant: "destructive"
      });
      return;
    }

    try {
      await AetherMindAPIService.saveJournalEntry({
        emotion: selectedEmotion.name,
        intensity: selectedEmotion.intensity,
        text: journalText,
        timestamp: new Date(),
        sentiment
      });

      toast({
        title: "Journal entry saved",
        description: "Your thoughts and feelings have been recorded safely",
      });

      // Reset form
      setSelectedEmotion(null);
      setJournalText('');
      setSentiment('neutral');
    } catch (error) {
      toast({
        title: "Save failed",
        description: "Please try again in a moment",
        variant: "destructive"
      });
    }
  };

  const getSentimentBg = () => {
    switch (sentiment) {
      case 'positive': return 'bg-success/5 border-success/20';
      case 'negative': return 'bg-emergency/5 border-emergency/20';
      default: return 'bg-muted/50 border-border';
    }
  };

  return (
    <Card className="p-6 space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-semibold text-foreground mb-2 flex items-center justify-center gap-2">
          <Heart className="w-6 h-6 text-secondary" />
          Emotion Journal
        </h2>
        <p className="text-muted-foreground">
          Express your feelings and track your emotional journey
        </p>
      </div>

      {/* Emotion Wheel */}
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-center">How are you feeling?</h3>
        <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
          {emotions.map((emotion) => (
            <button
              key={emotion.name}
              onClick={() => handleEmotionSelect(emotion)}
              className={`
                ${emotion.color} p-4 rounded-full text-white font-medium text-sm
                transition-all duration-200 hover:scale-105 hover:shadow-embrace
                ${selectedEmotion?.name === emotion.name ? 'ring-4 ring-primary scale-110' : ''}
              `}
            >
              {emotion.name}
            </button>
          ))}
        </div>
        
        {selectedEmotion && (
          <div className="text-center p-4 bg-primary/5 rounded-lg border border-primary/20">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Brain className="w-5 h-5 text-primary" />
              <span className="font-medium">You're feeling {selectedEmotion.name}</span>
            </div>
            <div className="w-full bg-muted rounded-full h-2">
              <div 
                className="bg-primary h-2 rounded-full transition-all duration-500"
                style={{ width: `${selectedEmotion.intensity * 100}%` }}
              />
            </div>
            <p className="text-sm text-muted-foreground mt-2">
              Intensity: {Math.round(selectedEmotion.intensity * 100)}%
            </p>
          </div>
        )}
      </div>

      {/* Journal Text Area */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium">Tell us more about it</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={handleVoiceInput}
            disabled={isListening}
            className="flex items-center gap-2"
          >
            <Mic className={`w-4 h-4 ${isListening ? 'animate-pulse text-emergency' : ''}`} />
            {isListening ? 'Listening...' : 'Voice'}
          </Button>
        </div>
        
        <Textarea
          value={journalText}
          onChange={(e) => {
            setJournalText(e.target.value);
            if (e.target.value.trim()) {
              handleTextAnalysis(e.target.value);
            }
          }}
          placeholder="Share what's on your mind... Your thoughts are safe here."
          className={`min-h-32 transition-all duration-300 ${getSentimentBg()}`}
          disabled={isAnalyzing}
        />
        
        {isAnalyzing && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Zap className="w-4 h-4 animate-pulse" />
            Analyzing your feelings...
          </div>
        )}
      </div>

      {/* Sentiment Indicator */}
      {sentiment !== 'neutral' && (
        <div className={`p-4 rounded-lg border-2 ${getSentimentBg()}`}>
          <div className="flex items-center gap-2 mb-2">
            <div className={`w-3 h-3 rounded-full ${
              sentiment === 'positive' ? 'bg-success' : 'bg-emergency'
            }`} />
            <span className="font-medium">
              {sentiment === 'positive' ? 'Positive sentiment detected' : 'Difficult feelings recognized'}
            </span>
          </div>
          <p className="text-sm text-muted-foreground">
            {sentiment === 'positive' 
              ? 'Your words reflect strength and hope' 
              : 'Remember: it\'s okay to feel this way. You\'re not alone.'
            }
          </p>
        </div>
      )}

      {/* Save Button */}
      <Button 
        onClick={handleSaveEntry}
        className="w-full"
        disabled={!selectedEmotion || !journalText.trim()}
      >
        <Send className="w-4 h-4 mr-2" />
        Save Journal Entry
      </Button>
    </Card>
  );
};

export default EmotionWheel;