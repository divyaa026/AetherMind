import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  Award, 
  Calendar, 
  BarChart3,
  Sparkles,
  Target,
  Zap
} from 'lucide-react';
import AetherMindAPIService, { Achievement, UserInsight } from '@/services/AetherMindAPI';

interface PetStage {
  name: string;
  emoji: string;
  threshold: number;
  message: string;
}

const GrowthCompanion: React.FC = () => {
  const [insights, setInsights] = useState<UserInsight | null>(null);
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [loading, setLoading] = useState(true);

  const petStages: PetStage[] = [
    { name: 'Seed', emoji: '🌱', threshold: 0, message: 'Your journey begins!' },
    { name: 'Sprout', emoji: '🌿', threshold: 10, message: 'Growing stronger!' },
    { name: 'Sapling', emoji: '🌳', threshold: 25, message: 'Reaching new heights!' },
    { name: 'Blossom', emoji: '🌸', threshold: 50, message: 'Beautiful growth!' },
    { name: 'Mighty Tree', emoji: '🌲', threshold: 100, message: 'Incredible resilience!' }
  ];

  const getCurrentPetStage = (progress: number): PetStage => {
    const stage = petStages
      .slice()
      .reverse()
      .find(stage => progress >= stage.threshold);
    return stage || petStages[0];
  };

  useEffect(() => {
    const loadData = async () => {
      try {
        const [insightsData, achievementsData] = await Promise.all([
          AetherMindAPIService.getUserInsights(),
          AetherMindAPIService.getAchievements()
        ]);
        setInsights(insightsData);
        setAchievements(achievementsData);
      } catch (error) {
        console.error('Failed to load growth data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return (
      <Card className="p-6 text-center">
        <div className="animate-pulse">
          <div className="text-6xl mb-4">🌱</div>
          <p className="text-muted-foreground">Loading your growth data...</p>
        </div>
      </Card>
    );
  }

  if (!insights) return null;

  const currentStage = getCurrentPetStage(insights.weeklyProgress * 100);
  const nextStage = petStages.find(stage => stage.threshold > insights.weeklyProgress * 100);

  return (
    <div className="space-y-6">
      {/* Growth Companion Pet */}
      <Card className="p-6 bg-gradient-growth shadow-growth">
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Sparkles className="w-6 h-6 text-success" />
            <h2 className="text-2xl font-semibold">Growth Companion</h2>
            <Sparkles className="w-6 h-6 text-success" />
          </div>
          
          {/* Pet Display */}
          <div className="relative">
            <div className="text-8xl animate-float">
              {currentStage.emoji}
            </div>
            <div className="absolute -top-2 -right-2 bg-success text-success-foreground rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold animate-pulse-gentle">
              {Math.round(insights.weeklyProgress * 100)}
            </div>
          </div>
          
          <div>
            <h3 className="text-xl font-semibold text-success-foreground mb-2">
              {currentStage.name}
            </h3>
            <p className="text-success-foreground/80 mb-4">
              {currentStage.message}
            </p>
            
            {nextStage && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm text-success-foreground/70">
                  <span>Progress to {nextStage.name}</span>
                  <span>{Math.round(insights.weeklyProgress * 100)}/{nextStage.threshold}</span>
                </div>
                <Progress 
                  value={(insights.weeklyProgress * 100 / nextStage.threshold) * 100}
                  className="h-2"
                />
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Weekly Insights */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-primary" />
          <h3 className="text-xl font-semibold">This Week's Insights</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Mood Trend Chart */}
          <div className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Mood Trend
            </h4>
            <div className="h-32 flex items-end gap-1">
              {insights.moodTrend.map((mood, index) => (
                <div 
                  key={index}
                  className="flex-1 bg-gradient-to-t from-primary to-primary-glow rounded-t"
                  style={{ height: `${mood * 100}%`, minHeight: '8px' }}
                />
              ))}
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map(day => (
                <span key={day}>{day}</span>
              ))}
            </div>
          </div>
          
          {/* Stats */}
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4 text-success" />
                <span className="text-sm">Activity Streak</span>
              </div>
              <Badge variant="secondary" className="bg-success text-success-foreground">
                {insights.activityStreak} days
              </Badge>
            </div>
            
            <div className="space-y-2">
              <h5 className="text-sm font-medium">Top Emotions</h5>
              {insights.topEmotions.map((emotion, index) => (
                <div key={emotion.emotion} className="flex items-center justify-between text-sm">
                  <span className="capitalize">{emotion.emotion}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary"
                        style={{ width: `${(emotion.count / insights.topEmotions[0].count) * 100}%` }}
                      />
                    </div>
                    <span className="text-muted-foreground">{emotion.count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Achievements */}
      <Card className="p-6">
        <div className="flex items-center gap-2 mb-4">
          <Award className="w-5 h-5 text-secondary" />
          <h3 className="text-xl font-semibold">Achievements</h3>
        </div>
        
        <div className="grid gap-4">
          {achievements.map((achievement) => (
            <div 
              key={achievement.id}
              className={`flex items-start gap-4 p-4 rounded-lg border transition-all duration-300 ${
                achievement.unlockedAt 
                  ? 'bg-secondary/5 border-secondary/20 shadow-embrace' 
                  : 'bg-muted/20 border-border'
              }`}
            >
              <div className="text-2xl">
                {achievement.icon}
              </div>
              
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h4 className={`font-medium ${achievement.unlockedAt ? 'text-secondary-foreground' : ''}`}>
                    {achievement.title}
                  </h4>
                  {achievement.unlockedAt ? (
                    <Badge className="bg-secondary text-secondary-foreground">
                      <Target className="w-3 h-3 mr-1" />
                      Unlocked
                    </Badge>
                  ) : (
                    <Badge variant="outline">
                      {Math.round(achievement.progress * 100)}%
                    </Badge>
                  )}
                </div>
                
                <p className="text-sm text-muted-foreground mb-3">
                  {achievement.description}
                </p>
                
                {!achievement.unlockedAt && (
                  <Progress 
                    value={achievement.progress * 100} 
                    className="h-2"
                  />
                )}
                
                {achievement.unlockedAt && (
                  <p className="text-xs text-secondary-foreground/70">
                    Unlocked {achievement.unlockedAt.toLocaleDateString()}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-6 text-center">
          <Button variant="outline" className="animate-pulse-gentle">
            <Zap className="w-4 h-4 mr-2" />
            View All Achievements
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default GrowthCompanion;