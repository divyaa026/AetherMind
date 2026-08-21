// AetherMind API Service - Comprehensive wellness platform API
// Production-ready with mock fallback for development

// ============ CORE TYPE DEFINITIONS ============

export interface CrisisAssessment {
  riskLevel: number;
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

// ============ DAILY CHECK-IN TYPES ============

export interface CheckInData {
  mood: number;
  energy: number;
  stress: number;
  journal: string;
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  timestamp: Date;
}

// ============ FORECAST TYPES ============

export interface EmotionalForecast {
  predictedMood: number;
  predictedEnergy: number;
  predictedStress: number;
  moodTrend: 'up' | 'down' | 'stable';
  energyTrend: 'up' | 'down' | 'stable';
  stressTrend: 'up' | 'down' | 'stable';
  recommendation: string;
  timeline: { date: string; mood: number; energy: number; stress: number }[];
}

export interface PatternInsight {
  title: string;
  description: string;
  type: 'positive' | 'negative' | 'neutral';
  stat?: string;
  icon: string;
  actionTab?: string;
}

export interface WeeklyHeatmap {
  data: number[][];
}

// ============ INTERVENTION TYPES ============

export interface Intervention {
  id: string;
  title: string;
  description: string;
  type: string;
  duration: string;
  tags: string[];
  navigateTo?: string;
}

export interface WeeklyChallenge {
  id: string;
  title: string;
  description: string;
  type: string;
  participants: number;
  duration: string;
  currentProgress: number;
  targetProgress: number;
  unit: string;
  reward: string;
  daysRemaining: number;
  joined: boolean;
  completed: boolean;
  collectiveProgress?: number;
}

export interface ContextPrompt {
  id: string;
  title: string;
  suggestion: string;
  context: string;
}

// ============ RESILIENCE PROGRAM TYPES ============

export interface ResilienceWeek {
  weekNumber: number;
  description: string;
  locked: boolean;
  lessons: ResilienceLesson[];
}

export interface ResilienceLesson {
  id: string;
  title: string;
  duration: string;
  type: 'video' | 'reading' | 'exercise';
  completed: boolean;
  locked: boolean;
  keyPoints?: string[];
  content?: string;
  exercise?: ResilienceExercise;
}

export interface ResilienceExercise {
  instructions: string;
  navigateTo: string;
}

// ============ ANALYTICS TYPES ============

export interface CorrelationData {
  factor1: string;
  factor2: string;
  value: number;
  description: string;
  insight: string;
  icon: string;
}

export interface ProgressReport {
  avgMood: number;
  avgEnergy: number;
  avgStress: number;
  moodTrend: 'up' | 'down' | 'stable';
  energyTrend: 'up' | 'down' | 'stable';
  stressTrend: 'up' | 'down' | 'stable';
  totalCheckins: number;
  weeklyRhythm: { day: string; avgMood: number; note?: string }[];
  seasonalPatterns: { season: string; avgMood: number; trend: string; insight: string }[];
  triggers: { name: string; impact: 'positive' | 'negative'; effect: string; metric: string }[];
}

export interface CustomGoal {
  id: string;
  title: string;
  description: string;
  icon: string;
  currentValue: number;
  targetValue: number;
  unit: string;
  startDate: string;
  targetDate: string;
  completed: boolean;
}

// ============ INTEGRATION TYPES ============

export interface IntegrationService {
  id: string;
  name: string;
  type: string;
  description: string;
  category: 'wearable' | 'productivity' | 'other';
  connected: boolean;
  lastSync?: string;
  dataTypes?: string[];
  latestData?: { steps?: number; sleepHours?: number; heartRate?: number };
  features?: { name: string; enabled: boolean }[];
}

// ============ COMMUNITY TYPES ============

export interface CommunityPost {
  id: string;
  username: string;
  avatar: string;
  content: string;
  timeAgo: string;
  likes: number;
  comments: number;
  liked: boolean;
  badge?: string;
  tags?: string[];
}

export interface GroupChallenge {
  id: string;
  title: string;
  description: string;
  participants: number;
  duration: string;
  joined: boolean;
  collectiveProgress: number;
}

export interface Resource {
  id: string;
  title: string;
  description: string;
  type: 'article' | 'podcast' | 'video';
  duration: string;
  author: string;
  rating: number;
}

export interface Expert {
  id: string;
  name: string;
  initials: string;
  credential: string;
  specialty: string;
  yearsExperience: number;
  totalQAs: number;
}

// ============ PROFESSIONAL SUPPORT TYPES ============

export interface Therapist {
  id: string;
  name: string;
  initials: string;
  credential: string;
  specialty: string;
  location: string;
  rating: number;
  reviewCount: number;
  availability: string;
  pricePerSession: number;
  acceptsInsurance: boolean;
  verified: boolean;
  sessionTypes?: string[];
}

export interface SessionNote {
  id: string;
  content: string;
  date: Date;
  discussed: boolean;
}

export interface CrisisResource {
  id: string;
  name: string;
  type: 'hotline' | 'local' | 'online';
  description?: string;
  number?: string;
  address?: string;
  availability: string;
}

// ============ PRIVACY TYPES ============

export interface PrivacySettings {
  privacyScore: number;
}

export interface DataCollection {
  id: string;
  name: string;
  description: string;
  purpose: string;
  icon: string;
  enabled: boolean;
  required: boolean;
  stats?: { records: number; size: string; lastUpdated: string };
}

export interface FederatedStatus {
  modelSize: string;
  lastTraining: string;
  dataPoints: number;
}

// ============ PERSONALIZATION TYPES ============

export interface UserPreferences {
  exerciseDuration: number;
  suggestSocialActivities: boolean;
  suggestOutdoorActivities: boolean;
  suggestDigitalDetox: boolean;
  focusAreas: string[];
  contentFormat: 'video' | 'reading' | 'mixed';
  contentDepth: 'practical' | 'balanced' | 'science';
  reminders: { morning: boolean; midday: boolean; evening: boolean };
}

export interface WellnessGoal {
  id: string;
  label: string;
  active: boolean;
}

// ============ GAMIFICATION TYPES ============

export interface Streak {
  current: number;
  longest: number;
  weekProgress: boolean[];
}

export interface GamificationBadge {
  id: string;
  name: string;
  description: string;
  icon: string;
  color: string;
  unlocked: boolean;
  unlockedAt?: string;
  requirement?: string;
  progress?: number;
}

export interface Milestone {
  id: string;
  title: string;
  description: string;
  current: number;
  target: number;
  completed: boolean;
  reward?: string;
}

class AetherMindAPIService {
  private static baseUrl = 'https://api.aethermind.ai';
  private static useMockData = true;

  // ============ CRISIS ANALYSIS ============
  
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

  private static async mockAnalyzeText(text: string): Promise<CrisisAssessment> {
    await new Promise(resolve => setTimeout(resolve, 800));
    const lowerText = text.toLowerCase();
    const crisisKeywords = ['hurt', 'die', 'kill', 'end it', 'suicide', 'harm'];
    const concernKeywords = ['sad', 'lonely', 'hopeless', 'worthless', 'tired'];
    const positiveKeywords = ['better', 'hopeful', 'grateful', 'improving', 'proud'];
    let riskLevel = 0.1;
    const flags: string[] = [];
    if (crisisKeywords.some(word => lowerText.includes(word))) {
      riskLevel = Math.max(riskLevel, 0.9);
      flags.push('crisis_language');
    }
    if (concernKeywords.some(word => lowerText.includes(word))) {
      riskLevel = Math.max(riskLevel, 0.6);
      flags.push('concerning_mood');
    }
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

  // ============ SAFETY MONITORING ============

  static createSafetyStream(): ReadableStream<SafetyStatus> {
    return new ReadableStream({
      start(controller) {
        const interval = setInterval(() => {
          const status: SafetyStatus = {
            riskLevel: Math.random() * 0.3 + (Math.random() > 0.9 ? 0.7 : 0),
            lastUpdate: new Date(),
            trend: ['improving', 'stable', 'concerning'][Math.floor(Math.random() * 3)] as 'improving' | 'stable' | 'concerning'
          };
          controller.enqueue(status);
        }, 5000);
        return () => clearInterval(interval);
      }
    });
  }

  // ============ DAILY CHECK-INS (localStorage) ============

  static async saveCheckIn(data: CheckInData): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 300));
    const existing = JSON.parse(localStorage.getItem('aethermind_checkins') || '[]');
    existing.push({ ...data, timestamp: new Date().toISOString() });
    localStorage.setItem('aethermind_checkins', JSON.stringify(existing));
  }

  static async getTodayCheckins(): Promise<CheckInData[]> {
    await new Promise(resolve => setTimeout(resolve, 200));
    const all = JSON.parse(localStorage.getItem('aethermind_checkins') || '[]');
    const today = new Date().toDateString();
    return all.filter((c: any) => new Date(c.timestamp).toDateString() === today)
      .map((c: any) => ({ ...c, timestamp: new Date(c.timestamp) }));
  }

  // ============ EMOTIONAL FORECAST ============

  static async getEmotionalForecast(days: number): Promise<EmotionalForecast> {
    await new Promise(resolve => setTimeout(resolve, 600));
    const timeline = Array.from({ length: Math.min(days, 7) }, (_, i) => ({
      date: new Date(Date.now() - i * 86400000).toISOString(),
      mood: Math.floor(Math.random() * 4) + 5,
      energy: Math.floor(Math.random() * 4) + 5,
      stress: Math.floor(Math.random() * 4) + 3
    })).reverse();
    return {
      predictedMood: 7, predictedEnergy: 6, predictedStress: 5,
      moodTrend: 'up', energyTrend: 'stable', stressTrend: 'down',
      recommendation: "Based on your patterns, tomorrow looks positive! Consider starting with your morning breathing exercise.",
      timeline
    };
  }

  static async getPatternInsights(): Promise<PatternInsight[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { title: "Mondays are challenging", description: "Your stress levels are typically 30% higher on Mondays", type: 'negative', stat: "+30% stress", icon: 'calendar', actionTab: 'breathing' },
      { title: "Exercise boosts your energy", description: "On days you exercise, your energy peaks by 2.3 points", type: 'positive', stat: "+2.3 energy", icon: 'activity' },
      { title: "Sleep impacts next-day stress", description: "When you sleep less than 6 hours, next day stress increases by 40%", type: 'negative', stat: "+40% stress", icon: 'moon' },
      { title: "Social connection helps", description: "Your mood stays elevated for 48 hours after social interactions", type: 'positive', stat: "48h mood lift", icon: 'target' }
    ];
  }

  static async getWeeklyHeatmap(): Promise<WeeklyHeatmap> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return { data: [[4, 6, 5, 7, 4, 3, 3], [5, 7, 6, 8, 5, 4, 4], [3, 5, 4, 6, 4, 3, 3]] };
  }

  // ============ INTERVENTIONS ============

  static async getMicroInterventions(): Promise<Intervention[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: '1', title: '4-7-8 Breathing Exercise', description: 'A calming breathing technique to reduce anxiety', type: 'breathing', duration: '2 min', tags: ['stress', 'anxiety'], navigateTo: 'breathing' },
      { id: '2', title: 'Gratitude Journaling', description: 'Write down 3 things you are grateful for today', type: 'gratitude', duration: '3 min', tags: ['mood', 'perspective'], navigateTo: 'journal' },
      { id: '3', title: '5-Minute Mindfulness', description: 'A quick mindfulness meditation to center yourself', type: 'mindfulness', duration: '5 min', tags: ['focus', 'calm'] },
      { id: '4', title: 'Connection Call', description: 'Reach out to a friend or family member', type: 'social', duration: '5 min', tags: ['social', 'support'] },
      { id: '5', title: 'Nature Walk', description: 'Take a short walk outside and notice 5 natural things', type: 'nature', duration: '5 min', tags: ['energy', 'mood'] }
    ];
  }

  static async getWeeklyChallenges(): Promise<WeeklyChallenge[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: '1', title: 'Digital Detox Evening', description: 'No screens after 8 PM for 5 days', type: 'digital-detox', participants: 1247, duration: '1 week', currentProgress: 3, targetProgress: 5, unit: 'days', reward: '50 points + Digital Detox badge', daysRemaining: 4, joined: true, completed: false, collectiveProgress: 67 },
      { id: '2', title: 'Social Connection Challenge', description: 'Have meaningful conversations with 3 different people', type: 'social', participants: 892, duration: '1 week', currentProgress: 1, targetProgress: 3, unit: 'connections', reward: '75 points + Social Butterfly badge', daysRemaining: 5, joined: false, completed: false },
      { id: '3', title: 'Sleep Consistency Week', description: 'Go to bed at the same time (±30 min) every night', type: 'sleep', participants: 2341, duration: '1 week', currentProgress: 0, targetProgress: 7, unit: 'nights', reward: '100 points + Sleep Master badge', daysRemaining: 7, joined: false, completed: false }
    ];
  }

  static async getContextPrompts(): Promise<ContextPrompt[]> {
    await new Promise(resolve => setTimeout(resolve, 200));
    const hour = new Date().getHours();
    const prompts: ContextPrompt[] = [];
    if (hour >= 9 && hour <= 17) prompts.push({ id: 'work-1', title: "You're at work", suggestion: "Time for a 2-minute stretch break?", context: 'work' });
    if (hour >= 22 || hour <= 2) prompts.push({ id: 'night-1', title: "Late night scrolling?", suggestion: "Consider starting your wind-down routine", context: 'late-night' });
    if (new Date().getDay() === 0 || new Date().getDay() === 6) prompts.push({ id: 'weekend-1', title: "Weekend alone?", suggestion: "Here are some social activity ideas", context: 'weekend' });
    return prompts;
  }

  static async completeIntervention(id: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 300));
    console.log('Intervention completed:', id);
  }

  // ============ RESILIENCE PROGRAM ============

  static async getResilienceProgram(): Promise<ResilienceWeek[]> {
    await new Promise(resolve => setTimeout(resolve, 500));
    return [
      { weekNumber: 1, description: "Learn to identify and name your emotions accurately", locked: false, lessons: [
        { id: 'w1-1', title: 'Understanding Your Emotional Landscape', duration: '5 min', type: 'video', completed: true, locked: false, keyPoints: ['Emotions are signals, not facts', 'Primary vs secondary emotions', 'The emotion-thought connection'] },
        { id: 'w1-2', title: 'The Emotion Wheel Exercise', duration: '10 min', type: 'exercise', completed: true, locked: false, keyPoints: ['Identifying nuanced emotions', 'Moving beyond good and bad'], exercise: { instructions: 'Use the emotion wheel to identify exactly what you are feeling right now', navigateTo: 'journal' } },
        { id: 'w1-3', title: 'Body Awareness & Emotions', duration: '7 min', type: 'reading', completed: false, locked: false, content: 'Emotions manifest in our bodies before we consciously recognize them.' }
      ]},
      { weekNumber: 2, description: "Deepen your emotional awareness through daily practice", locked: false, lessons: [
        { id: 'w2-1', title: 'Mindful Emotion Observation', duration: '6 min', type: 'video', completed: false, locked: false },
        { id: 'w2-2', title: 'Trigger Mapping Exercise', duration: '15 min', type: 'exercise', completed: false, locked: false }
      ]},
      { weekNumber: 3, description: "Learn to modulate your stress response", locked: true, lessons: [{ id: 'w3-1', title: 'The Stress Response System', duration: '8 min', type: 'video', completed: false, locked: true }] },
      { weekNumber: 4, description: "Advanced stress management techniques", locked: true, lessons: [] },
      { weekNumber: 5, description: "Develop cognitive flexibility skills", locked: true, lessons: [] },
      { weekNumber: 6, description: "Practice reframing and perspective-taking", locked: true, lessons: [] },
      { weekNumber: 7, description: "Integrate all skills into daily life", locked: true, lessons: [] },
      { weekNumber: 8, description: "Build your personal resilience toolkit", locked: true, lessons: [] }
    ];
  }

  static async getResilienceScore(): Promise<number> {
    await new Promise(resolve => setTimeout(resolve, 200));
    return 68;
  }

  static async completeResilienceLesson(lessonId: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 400));
    console.log('Lesson completed:', lessonId);
  }

  // ============ ANALYTICS ============

  static async getCorrelations(days: number): Promise<CorrelationData[]> {
    await new Promise(resolve => setTimeout(resolve, 500));
    return [
      { factor1: 'Sleep', factor2: 'Mood', value: 0.62, description: 'Strong positive correlation', insight: 'Better sleep quality consistently leads to improved mood', icon: 'moon' },
      { factor1: 'Exercise', factor2: 'Energy', value: 0.54, description: 'Moderate positive correlation', insight: 'On days you exercise, your energy levels are 2.3 points higher', icon: 'activity' },
      { factor1: 'Social Time', factor2: 'Mood', value: 0.48, description: 'Moderate positive correlation', insight: 'Social interactions provide a mood boost that lasts up to 48 hours', icon: 'users' },
      { factor1: 'Screen Time', factor2: 'Sleep', value: -0.41, description: 'Negative correlation', insight: 'Evening screen time negatively impacts your sleep quality', icon: 'clock' }
    ];
  }

  static async getProgressReport(days: number): Promise<ProgressReport> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return {
      avgMood: 6.8, avgEnergy: 6.2, avgStress: 4.5, moodTrend: 'up', energyTrend: 'stable', stressTrend: 'down', totalCheckins: 28,
      weeklyRhythm: [
        { day: 'Monday', avgMood: 5.2, note: 'Challenging start' }, { day: 'Tuesday', avgMood: 6.1 }, { day: 'Wednesday', avgMood: 6.8, note: 'Peak stress day' },
        { day: 'Thursday', avgMood: 6.5 }, { day: 'Friday', avgMood: 7.2 }, { day: 'Saturday', avgMood: 7.8, note: 'Best day' }, { day: 'Sunday', avgMood: 7.4 }
      ],
      seasonalPatterns: [
        { season: 'Winter', avgMood: 5.8, trend: 'down', insight: 'Consider light therapy' }, { season: 'Spring', avgMood: 7.2, trend: 'up', insight: 'Your best season' },
        { season: 'Summer', avgMood: 7.0, trend: 'stable', insight: 'Stay hydrated' }, { season: 'Fall', avgMood: 6.4, trend: 'down', insight: 'Prepare for transition' }
      ],
      triggers: [
        { name: 'Work meetings', impact: 'negative', effect: '+15%', metric: 'stress' }, { name: 'Morning exercise', impact: 'positive', effect: '+2.3', metric: 'energy' },
        { name: 'Poor sleep', impact: 'negative', effect: '-1.5', metric: 'mood' }, { name: 'Nature time', impact: 'positive', effect: '-20%', metric: 'stress' }
      ]
    };
  }

  static async getCustomGoals(): Promise<CustomGoal[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return [
      { id: '1', title: 'Reduce Work Stress', description: 'Lower average work-day stress to below 5', icon: 'brain', currentValue: 5.8, targetValue: 5.0, unit: 'stress score', startDate: '2026-01-01', targetDate: '2026-02-01', completed: false },
      { id: '2', title: 'Morning Energy Boost', description: 'Achieve average morning energy of 7+', icon: 'activity', currentValue: 6.2, targetValue: 7.0, unit: 'energy score', startDate: '2026-01-01', targetDate: '2026-01-31', completed: false }
    ];
  }

  // ============ INTEGRATIONS ============

  static async getIntegrations(): Promise<IntegrationService[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: 'apple-health', name: 'Apple Health', type: 'apple-health', description: 'Sync health data from your iPhone and Apple Watch', category: 'wearable', connected: true, lastSync: '5 minutes ago', dataTypes: ['steps', 'sleep', 'heart-rate', 'hrv'], latestData: { steps: 8432, sleepHours: 7.2, heartRate: 68 } },
      { id: 'google-fit', name: 'Google Fit', type: 'google-fit', description: 'Connect your Android fitness data', category: 'wearable', connected: false },
      { id: 'fitbit', name: 'Fitbit', type: 'fitbit', description: 'Sync your Fitbit activity and sleep data', category: 'wearable', connected: false },
      { id: 'oura', name: 'Oura Ring', type: 'oura', description: 'Advanced sleep and readiness tracking', category: 'wearable', connected: false },
      { id: 'google-calendar', name: 'Google Calendar', type: 'calendar', description: 'Stress prediction based on your schedule', category: 'productivity', connected: true, features: [{ name: 'Predict stress around meetings', enabled: true }, { name: 'Suggest breaks between events', enabled: true }, { name: 'Track busy vs free time ratio', enabled: false }] },
      { id: 'outlook', name: 'Outlook Calendar', type: 'calendar', description: 'Sync your work calendar for insights', category: 'productivity', connected: false }
    ];
  }

  static async connectIntegration(serviceId: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log('Connected integration:', serviceId);
  }

  static async disconnectIntegration(serviceId: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 500));
    console.log('Disconnected integration:', serviceId);
  }

  static async syncIntegration(serviceId: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 2000));
    console.log('Synced integration:', serviceId);
  }

  static async exportData(format: 'csv' | 'json' | 'pdf'): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 1500));
    console.log('Exported data in format:', format);
  }

  // ============ COMMUNITY ============

  static async getCommunityPosts(): Promise<CommunityPost[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: '1', username: 'WellnessWarrior', avatar: '🌟', content: 'Just completed my 30-day mindfulness streak! The difference in my stress levels is incredible.', timeAgo: '2 hours ago', likes: 47, comments: 12, liked: false, badge: '30-Day Streak', tags: ['mindfulness', 'streak', 'progress'] },
      { id: '2', username: 'CalmSeeker', avatar: '🧘', content: 'Tip that helped me: Instead of fighting anxious thoughts, I now acknowledge them and let them pass like clouds.', timeAgo: '5 hours ago', likes: 89, comments: 23, liked: true, tags: ['anxiety', 'tips', 'mindset'] },
      { id: '3', username: 'GrowthMindset', avatar: '🌱', content: 'Remember: Your journey is unique. Comparing your chapter 1 to someone else\'s chapter 20 isn\'t fair to yourself. 💚', timeAgo: '1 day ago', likes: 156, comments: 34, liked: false, tags: ['motivation', 'selfcare'] }
    ];
  }

  static async getGroupChallenges(): Promise<GroupChallenge[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return [
      { id: '1', title: 'January Mindfulness Challenge', description: 'Complete 10 minutes of mindfulness daily for the month', participants: 3421, duration: 'January 2026', joined: true, collectiveProgress: 72 },
      { id: '2', title: 'Gratitude Chain', description: 'Share one thing you\'re grateful for each day', participants: 2156, duration: 'Ongoing', joined: false, collectiveProgress: 0 }
    ];
  }

  static async getResources(): Promise<Resource[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: '1', title: 'Understanding Anxiety: A Beginner\'s Guide', description: 'Learn the science behind anxiety and practical coping strategies', type: 'article', duration: '8 min read', author: 'Dr. Sarah Chen', rating: 4.8 },
      { id: '2', title: 'The Calm Mind Podcast', description: 'Weekly conversations about mental wellness and resilience', type: 'podcast', duration: '35 min', author: 'Mind Matters', rating: 4.9 },
      { id: '3', title: 'Guided Progressive Muscle Relaxation', description: 'Step-by-step video guide to release physical tension', type: 'video', duration: '15 min', author: 'Dr. Michael Torres', rating: 4.7 }
    ];
  }

  static async getExperts(): Promise<Expert[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return [
      { id: '1', name: 'Dr. Sarah Chen', initials: 'SC', credential: 'Ph.D., Licensed Psychologist', specialty: 'Anxiety & Stress Management', yearsExperience: 15, totalQAs: 234 },
      { id: '2', name: 'Dr. Michael Torres', initials: 'MT', credential: 'Psy.D., LMFT', specialty: 'Relationships & Trauma', yearsExperience: 12, totalQAs: 189 }
    ];
  }

  static async joinChallenge(challengeId: string): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 400));
    console.log('Joined challenge:', challengeId);
  }

  // ============ PROFESSIONAL SUPPORT ============

  static async getTherapists(): Promise<Therapist[]> {
    await new Promise(resolve => setTimeout(resolve, 500));
    return [
      { id: '1', name: 'Dr. Emily Richards', initials: 'ER', credential: 'Ph.D., Licensed Psychologist', specialty: 'Anxiety, Depression, Stress Management', location: 'New York, NY (Virtual available)', rating: 4.9, reviewCount: 127, availability: 'Next available: Tomorrow', pricePerSession: 180, acceptsInsurance: true, verified: true, sessionTypes: ['video', 'in-person'] },
      { id: '2', name: 'Dr. James Park', initials: 'JP', credential: 'Psy.D., LCSW', specialty: 'Trauma, PTSD, Life Transitions', location: 'Los Angeles, CA (Virtual only)', rating: 4.8, reviewCount: 89, availability: 'Next available: Friday', pricePerSession: 150, acceptsInsurance: true, verified: true, sessionTypes: ['video', 'chat'] }
    ];
  }

  static async getSessionNotes(): Promise<SessionNote[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return [
      { id: '1', content: 'Discuss recent work stress and deadline anxiety', date: new Date(Date.now() - 2 * 86400000), discussed: false },
      { id: '2', content: 'Follow up on sleep hygiene strategies from last session', date: new Date(Date.now() - 5 * 86400000), discussed: true }
    ];
  }

  static async addSessionNote(note: Omit<SessionNote, 'id' | 'discussed'>): Promise<SessionNote> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return { ...note, id: Date.now().toString(), discussed: false };
  }

  static async getCrisisResources(): Promise<CrisisResource[]> {
    await new Promise(resolve => setTimeout(resolve, 200));
    return [
      { id: '1', name: '988 Suicide & Crisis Lifeline', type: 'hotline', description: '24/7 free, confidential support', number: '988', availability: '24/7' },
      { id: '2', name: 'Crisis Text Line', type: 'hotline', description: 'Text HOME to 741741', number: '741741', availability: '24/7' },
      { id: '3', name: 'Community Mental Health Center', type: 'local', address: '123 Main St, Your City', availability: 'Mon-Fri 9am-5pm' }
    ];
  }

  // ============ PRIVACY ============

  static async getPrivacySettings(): Promise<PrivacySettings> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return { privacyScore: 92 };
  }

  static async getDataCollection(): Promise<DataCollection[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: 'checkins', name: 'Check-in Data', description: 'Your daily mood, energy, and stress ratings', purpose: 'Track patterns and provide personalized insights', icon: 'smartphone', enabled: true, required: true, stats: { records: 156, size: '24 KB', lastUpdated: 'Today' } },
      { id: 'journal', name: 'Journal Entries', description: 'Your written reflections and notes', purpose: 'Sentiment analysis and pattern detection', icon: 'file-text', enabled: true, required: false, stats: { records: 42, size: '18 KB', lastUpdated: 'Yesterday' } },
      { id: 'location', name: 'Location Context', description: 'General location for context-aware suggestions', purpose: 'Provide location-based wellness prompts', icon: 'eye', enabled: false, required: false },
      { id: 'analytics', name: 'Usage Analytics', description: 'How you interact with the app', purpose: 'Improve app experience and features', icon: 'database', enabled: true, required: false }
    ];
  }

  static async getFederatedLearningStatus(): Promise<FederatedStatus> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return { modelSize: '2.4 MB', lastTraining: '2 hours ago', dataPoints: 156 };
  }

  static async updatePrivacyPermission(permissionId: string, enabled: boolean): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 400));
    console.log('Updated permission:', permissionId, enabled);
  }

  static async deleteAllData(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 2000));
    console.log('All data deleted');
  }

  static async exportAllData(): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 1500));
    console.log('All data exported');
  }

  // ============ PERSONALIZATION (localStorage) ============

  static async getUserPreferences(): Promise<UserPreferences> {
    await new Promise(resolve => setTimeout(resolve, 200));
    const stored = localStorage.getItem('aethermind_preferences');
    if (stored) return JSON.parse(stored);
    return {
      exerciseDuration: 5, suggestSocialActivities: true, suggestOutdoorActivities: true, suggestDigitalDetox: true,
      focusAreas: ['stress', 'sleep'], contentFormat: 'mixed', contentDepth: 'balanced',
      reminders: { morning: true, midday: false, evening: true }
    };
  }

  static async saveUserPreferences(preferences: UserPreferences): Promise<void> {
    await new Promise(resolve => setTimeout(resolve, 200));
    localStorage.setItem('aethermind_preferences', JSON.stringify(preferences));
  }

  static async getWellnessGoals(): Promise<WellnessGoal[]> {
    await new Promise(resolve => setTimeout(resolve, 200));
    return [
      { id: 'stress', label: 'Reduce work stress', active: true },
      { id: 'sleep', label: 'Better sleep quality', active: true }
    ];
  }

  // ============ GAMIFICATION (localStorage) ============

  static async getStreak(): Promise<Streak> {
    await new Promise(resolve => setTimeout(resolve, 100));
    const stored = localStorage.getItem('aethermind_streak');
    if (stored) return JSON.parse(stored);
    return { current: 0, longest: 0, weekProgress: [false, false, false, false, false, false, false] };
  }

  static async updateStreak(): Promise<void> {
    const streak = await this.getStreak();
    const today = new Date().getDay();
    streak.weekProgress[today] = true;
    streak.current = streak.weekProgress.filter(Boolean).length;
    streak.longest = Math.max(streak.longest, streak.current);
    localStorage.setItem('aethermind_streak', JSON.stringify(streak));
  }

  static async getBadges(): Promise<GamificationBadge[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: '1', name: 'First Steps', description: 'Complete your first check-in', icon: 'star', color: 'from-yellow-400 to-orange-500', unlocked: true, unlockedAt: '2026-01-01' },
      { id: '2', name: '7-Day Streak', description: 'Check in 7 days in a row', icon: 'flame', color: 'from-orange-400 to-red-500', unlocked: true, unlockedAt: '2026-01-08' },
      { id: '3', name: 'Mindfulness Master', description: 'Complete 20 breathing exercises', icon: 'brain', color: 'from-purple-400 to-indigo-500', unlocked: true, unlockedAt: '2026-01-10' },
      { id: '4', name: 'Energy Improver', description: 'Increase average energy by 2 points', icon: 'zap', color: 'from-blue-400 to-cyan-500', unlocked: false, requirement: 'Increase energy from 5.2 to 7.2', progress: 0.6 },
      { id: '5', name: 'Stress Warrior', description: 'Reduce average stress below 4', icon: 'shield', color: 'from-green-400 to-emerald-500', unlocked: false, requirement: 'Current: 5.2 → Target: 4.0', progress: 0.4 },
      { id: '6', name: '30-Day Champion', description: 'Maintain a 30-day check-in streak', icon: 'crown', color: 'from-amber-400 to-yellow-500', unlocked: false, requirement: '5/30 days completed', progress: 0.17 }
    ];
  }

  static async getMilestones(): Promise<Milestone[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return [
      { id: '1', title: '30-Day Journey', description: 'Complete 30 days of check-ins', current: 15, target: 30, completed: false, reward: '100 points + Special badge' },
      { id: '2', title: 'Resilience Week 1', description: 'Complete all Week 1 lessons', current: 2, target: 3, completed: false, reward: '50 points' },
      { id: '3', title: 'First Breathing Session', description: 'Complete your first breathing exercise', current: 1, target: 1, completed: true, reward: '10 points' }
    ];
  }

  static async getTotalPoints(): Promise<number> {
    await new Promise(resolve => setTimeout(resolve, 100));
    return 485;
  }

  static async getLevel(): Promise<{ current: number; name: string; pointsToNext: number }> {
    await new Promise(resolve => setTimeout(resolve, 100));
    return { current: 3, name: 'Wellness Explorer', pointsToNext: 115 };
  }

  // ============ EMOTION JOURNAL ============

  static async saveJournalEntry(entry: Omit<EmotionEntry, 'id'>): Promise<EmotionEntry> {
    await new Promise(resolve => setTimeout(resolve, 500));
    return { ...entry, id: Date.now().toString() };
  }

  static async getJournalEntries(): Promise<EmotionEntry[]> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return [
      { id: '1', emotion: 'hopeful', intensity: 0.7, text: 'Today felt a bit better than yesterday', timestamp: new Date(Date.now() - 86400000), sentiment: 'positive' },
      { id: '2', emotion: 'anxious', intensity: 0.8, text: 'Worried about tomorrow', timestamp: new Date(Date.now() - 172800000), sentiment: 'negative' }
    ];
  }

  // ============ GROWTH & ACHIEVEMENTS ============

  static async getUserInsights(): Promise<UserInsight> {
    await new Promise(resolve => setTimeout(resolve, 600));
    return {
      moodTrend: [0.4, 0.3, 0.5, 0.6, 0.4, 0.7, 0.8],
      activityStreak: 5,
      topEmotions: [{ emotion: 'hopeful', count: 12 }, { emotion: 'anxious', count: 8 }, { emotion: 'calm', count: 6 }],
      weeklyProgress: 0.75
    };
  }

  static async getAchievements(): Promise<Achievement[]> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return [
      { id: '1', title: 'First Steps', description: 'Completed your first breathing exercise', icon: '🌱', unlockedAt: new Date(Date.now() - 172800000), progress: 1 },
      { id: '2', title: 'Weekly Warrior', description: 'Journal for 7 consecutive days', icon: '💪', progress: 0.6 },
      { id: '3', title: 'Mindful Master', description: 'Complete 50 breathing sessions', icon: '🧘', progress: 0.14 }
    ];
  }

  // ============ EMERGENCY CONTACTS ============

  static getEmergencyContacts() {
    return [
      { name: 'National Suicide Prevention Lifeline', number: '988', available: '24/7', type: 'crisis' },
      { name: 'Crisis Text Line', number: 'Text HOME to 741741', available: '24/7', type: 'crisis' },
      { name: 'SAMHSA National Helpline', number: '1-800-662-4357', available: '24/7', type: 'support' }
    ];
  }
}

export default AetherMindAPIService;