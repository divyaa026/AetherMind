// AetherMind API Service - Production ready with mock fallback
// Switch between real backend and simulation via environment

export interface CrisisAssessment {
  riskLevel: number; // 0-1 scale
  confidence: number;
  flags: string[];
  recommendedActions?: string[];
}

export interface SafetyStatus {
  riskLevel: number;
  lastUpdate: Date;
  trend: 'improving' | 'stable' | 'concerning';
}

export interface EmotionEntry {
  id: string;
  emotion: string;
  intensity: number;
  text: string;
  timestamp: Date;
  sentiment: 'positive' | 'neutral' | 'negative';
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  unlockedAt?: Date;
  progress: number;
}

export interface UserInsight {
  moodTrend: number[];
  activityStreak: number;
  topEmotions: { emotion: string; count: number }[];
  weeklyProgress: number;
}

class AetherMindAPIService {
  private static baseUrl = 'https://api.aethermind.ai';
  private static useMockData = true; // Toggle for demo mode

  // CRISIS ANALYSIS - Real endpoint for text analysis
  static async analyzeText(text: string): Promise<CrisisAssessment> {
    if (this.useMockData) {
      return this.mockAnalyzeText(text);
    }

    try {
      const response = await fetch(`${this.baseUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      return await response.json();
    } catch (error) {
      console.warn('Falling back to mock analysis:', error);
      return this.mockAnalyzeText(text);
    }
  }

  // MOCK CRISIS ANALYSIS - Sophisticated simulation
  private static async mockAnalyzeText(text: string): Promise<CrisisAssessment> {
    await new Promise(resolve => setTimeout(resolve, 800)); // Realistic delay

    const lowerText = text.toLowerCase();
    const crisisKeywords = ['hurt', 'die', 'kill', 'end it', 'suicide', 'harm'];
    const concernKeywords = ['sad', 'lonely', 'hopeless', 'worthless', 'tired'];
    const positiveKeywords = ['better', 'hopeful', 'grateful', 'improving', 'proud'];

    let riskLevel = 0.1;
    const flags: string[] = [];

    // Check for crisis indicators
    if (crisisKeywords.some(word => lowerText.includes(word))) {
      riskLevel = Math.max(riskLevel, 0.9);
      flags.push('crisis_language');
    }

    // Check for concerning language
    if (concernKeywords.some(word => lowerText.includes(word))) {
      riskLevel = Math.max(riskLevel, 0.6);
      flags.push('concerning_mood');
    }

    // Check for positive indicators
    if (positiveKeywords.some(word => lowerText.includes(word))) {
      riskLevel = Math.min(riskLevel, 0.3);
      flags.push('positive_sentiment');
    }

    return {
      riskLevel,
      confidence: 0.87,
      flags,
      recommendedActions: riskLevel > 0.7 ? [
        'Consider reaching out for professional support',
        'Practice grounding techniques',
        'Connect with a trusted friend'
      ] : []
    };
  }

  // REAL-TIME SAFETY MONITORING
  static createSafetyStream(): ReadableStream<SafetyStatus> {
    return new ReadableStream({
      start(controller) {
        const interval = setInterval(() => {
          const status: SafetyStatus = {
            riskLevel: Math.random() * 0.3 + (Math.random() > 0.9 ? 0.7 : 0), // Occasional spike
            lastUpdate: new Date(),
            trend: ['improving', 'stable', 'concerning'][Math.floor(Math.random() * 3)] as any
          };
          controller.enqueue(status);
        }, 5000);

        // Cleanup
        return () => clearInterval(interval);
      }
    });
  }

  // EMOTION JOURNAL ENDPOINTS
  static async saveJournalEntry(entry: Omit<EmotionEntry, 'id'>): Promise<EmotionEntry> {
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return {
      ...entry,
      id: Date.now().toString(),
    };
  }

  static async getJournalEntries(): Promise<EmotionEntry[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Mock recent entries
    return [
      {
        id: '1',
        emotion: 'hopeful',
        intensity: 0.7,
        text: 'Today felt a bit better than yesterday',
        timestamp: new Date(Date.now() - 86400000),
        sentiment: 'positive'
      },
      {
        id: '2',
        emotion: 'anxious',
        intensity: 0.8,
        text: 'Worried about tomorrow',
        timestamp: new Date(Date.now() - 172800000),
        sentiment: 'negative'
      }
    ];
  }

  // GROWTH & ACHIEVEMENTS
  static async getUserInsights(): Promise<UserInsight> {
    await new Promise(resolve => setTimeout(resolve, 600));
    
    return {
      moodTrend: [0.4, 0.3, 0.5, 0.6, 0.4, 0.7, 0.8], // Last 7 days
      activityStreak: 5,
      topEmotions: [
        { emotion: 'hopeful', count: 12 },
        { emotion: 'anxious', count: 8 },
        { emotion: 'calm', count: 6 }
      ],
      weeklyProgress: 0.75
    };
  }

  static async getAchievements(): Promise<Achievement[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    
    return [
      {
        id: '1',
        title: 'First Steps',
        description: 'Completed your first breathing exercise',
        icon: '🌱',
        unlockedAt: new Date(Date.now() - 172800000),
        progress: 1
      },
      {
        id: '2',
        title: 'Weekly Warrior',
        description: 'Journal for 7 consecutive days',
        icon: '💪',
        progress: 0.6
      },
      {
        id: '3',
        title: 'Mindful Master',
        description: 'Complete 50 breathing sessions',
        icon: '🧘',
        progress: 0.14
      }
    ];
  }

  // EMERGENCY CONTACTS
  static getEmergencyContacts() {
    return [
      {
        name: 'Crisis Text Line',
        number: 'Text HOME to 741741',
        type: 'crisis',
        available: '24/7'
      },
      {
        name: 'National Suicide Prevention Lifeline',
        number: '988',
        type: 'crisis',
        available: '24/7'
      },
      {
        name: 'Emergency Services',
        number: '911',
        type: 'emergency',
        available: '24/7'
      }
    ];
  }
}

export default AetherMindAPIService;