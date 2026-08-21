import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import MockDataBanner from '@/components/MockDataBanner';
import {
  Flame,
  Trophy,
  Star,
  Award,
  Target,
  Calendar,
  TrendingUp,
  Sparkles,
  Gift,
  Medal,
  Crown,
  Heart,
  Zap,
  Shield,
  Moon,
  Sun,
  Brain,
  CheckCircle,
  Lock,
  Share2,
  ChevronRight
} from 'lucide-react';
import AetherMindAPIService, { Streak, GamificationBadge, Milestone } from '@/services/AetherMindAPI';

interface WellnessJourneyProps {
  onNavigate?: (tab: string) => void;
}

const WellnessJourney: React.FC<WellnessJourneyProps> = ({ onNavigate }) => {
  const [streak, setStreak] = useState<Streak | null>(null);
  const [badges, setBadges] = useState<GamificationBadge[]>([]);
  const [milestones, setMilestones] = useState<Milestone[]>([]);
  const [totalPoints, setTotalPoints] = useState(0);
  const [level, setLevel] = useState({ current: 1, name: 'Beginner', pointsToNext: 100 });
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadGamificationData();
  }, []);

  const loadGamificationData = async () => {
    setLoading(true);
    try {
      const [streakData, badgesData, milestonesData, pointsData, levelData] = await Promise.all([
        AetherMindAPIService.getStreak(),
        AetherMindAPIService.getBadges(),
        AetherMindAPIService.getMilestones(),
        AetherMindAPIService.getTotalPoints(),
        AetherMindAPIService.getLevel()
      ]);
      setStreak(streakData);
      setBadges(badgesData);
      setMilestones(milestonesData);
      setTotalPoints(pointsData);
      setLevel(levelData);
    } catch (error) {
      console.error('Failed to load gamification data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getBadgeIcon = (iconName: string) => {
    const icons: Record<string, React.ReactNode> = {
      'flame': <Flame className="w-6 h-6" />,
      'star': <Star className="w-6 h-6" />,
      'trophy': <Trophy className="w-6 h-6" />,
      'award': <Award className="w-6 h-6" />,
      'medal': <Medal className="w-6 h-6" />,
      'crown': <Crown className="w-6 h-6" />,
      'heart': <Heart className="w-6 h-6" />,
      'zap': <Zap className="w-6 h-6" />,
      'shield': <Shield className="w-6 h-6" />,
      'moon': <Moon className="w-6 h-6" />,
      'sun': <Sun className="w-6 h-6" />,
      'brain': <Brain className="w-6 h-6" />,
      'target': <Target className="w-6 h-6" />,
    };
    return icons[iconName] || <Star className="w-6 h-6" />;
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-muted rounded w-1/3"></div>
          <div className="h-48 bg-muted rounded"></div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <MockDataBanner feature="Wellness Journey" storageType="localStorage" />
      
      {/* Header with Level */}
      <Card className="p-6 bg-gradient-to-br from-amber-500/10 via-primary/5 to-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-14 h-14 rounded-full bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
              <Crown className="w-7 h-7 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Wellness Journey</h2>
              <div className="flex items-center gap-2">
                <Badge className="bg-amber-500/20 text-amber-700">
                  Level {level.current}: {level.name}
                </Badge>
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">{totalPoints}</div>
            <div className="text-xs text-muted-foreground">Total Points</div>
          </div>
        </div>

        {/* Level Progress */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Progress to Level {level.current + 1}</span>
            <span className="font-medium">{level.pointsToNext} pts to go</span>
          </div>
          <Progress value={70} className="h-3" />
        </div>
      </Card>

      {/* Streak Card */}
      {streak && (
        <Card className={`p-6 ${streak.current > 0 ? 'bg-gradient-to-br from-orange-500/10 to-red-500/5' : ''}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                streak.current > 0 
                  ? 'bg-gradient-to-br from-orange-500 to-red-500' 
                  : 'bg-muted'
              }`}>
                <Flame className={`w-8 h-8 ${streak.current > 0 ? 'text-white' : 'text-muted-foreground'}`} />
              </div>
              <div>
                <div className="text-3xl font-bold">
                  {streak.current} days
                </div>
                <div className="text-sm text-muted-foreground">
                  Current streak
                </div>
              </div>
            </div>
            
            <div className="text-right">
              <div className="flex items-center gap-2 mb-1">
                <Trophy className="w-4 h-4 text-amber-500" />
                <span className="font-semibold">{streak.longest} days</span>
              </div>
              <div className="text-xs text-muted-foreground">Best streak</div>
            </div>
          </div>

          {/* Weekly View */}
          <div className="mt-6 flex justify-between">
            {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((day, idx) => {
              const isCompleted = streak.weekProgress?.[idx];
              const isToday = idx === new Date().getDay() - 1 || (idx === 6 && new Date().getDay() === 0);
              
              return (
                <div 
                  key={idx}
                  className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-medium ${
                    isCompleted 
                      ? 'bg-success text-success-foreground' 
                      : isToday 
                        ? 'bg-primary/20 text-primary border-2 border-primary' 
                        : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {isCompleted ? <CheckCircle className="w-5 h-5" /> : day}
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">
            <Star className="w-4 h-4 mr-2" />
            Badges
          </TabsTrigger>
          <TabsTrigger value="milestones">
            <Trophy className="w-4 h-4 mr-2" />
            Milestones
          </TabsTrigger>
          <TabsTrigger value="progress">
            <TrendingUp className="w-4 h-4 mr-2" />
            Progress
          </TabsTrigger>
        </TabsList>

        {/* Badges Tab */}
        <TabsContent value="overview" className="space-y-4 mt-4">
          {/* Earned Badges */}
          <div>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Award className="w-5 h-5 text-amber-500" />
              Earned Badges ({badges.filter(b => b.unlocked).length})
            </h3>
            
            <div className="grid grid-cols-3 gap-3">
              {badges.filter(b => b.unlocked).map((badge) => (
                <Card 
                  key={badge.id}
                  className="p-4 text-center hover:shadow-lg transition-shadow cursor-pointer"
                >
                  <div className={`w-12 h-12 rounded-full mx-auto mb-2 flex items-center justify-center bg-gradient-to-br ${badge.color}`}>
                    {getBadgeIcon(badge.icon)}
                  </div>
                  <div className="font-medium text-sm">{badge.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">{badge.description}</div>
                  {badge.unlockedAt && (
                    <Badge variant="outline" className="mt-2 text-xs">
                      {new Date(badge.unlockedAt).toLocaleDateString()}
                    </Badge>
                  )}
                </Card>
              ))}
            </div>
          </div>

          {/* Locked Badges */}
          <div>
            <h3 className="font-semibold mb-3 flex items-center gap-2">
              <Lock className="w-5 h-5 text-muted-foreground" />
              Badges to Unlock ({badges.filter(b => !b.unlocked).length})
            </h3>
            
            <div className="grid grid-cols-3 gap-3">
              {badges.filter(b => !b.unlocked).map((badge) => (
                <Card 
                  key={badge.id}
                  className="p-4 text-center opacity-60"
                >
                  <div className="w-12 h-12 rounded-full mx-auto mb-2 flex items-center justify-center bg-muted">
                    <Lock className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div className="font-medium text-sm">{badge.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">{badge.requirement}</div>
                  {badge.progress !== undefined && (
                    <Progress value={badge.progress * 100} className="h-1 mt-2" />
                  )}
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        {/* Milestones Tab */}
        <TabsContent value="milestones" className="space-y-4 mt-4">
          {milestones.map((milestone, idx) => (
            <Card 
              key={milestone.id}
              className={`p-4 ${milestone.completed ? 'bg-success/5 border-success/20' : ''}`}
            >
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                  milestone.completed 
                    ? 'bg-success/20 text-success' 
                    : 'bg-muted text-muted-foreground'
                }`}>
                  {milestone.completed ? (
                    <CheckCircle className="w-6 h-6" />
                  ) : (
                    <span className="font-bold">{idx + 1}</span>
                  )}
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h4 className="font-semibold">{milestone.title}</h4>
                    {milestone.completed && (
                      <Badge className="bg-success text-success-foreground">
                        Achieved!
                      </Badge>
                    )}
                  </div>
                  
                  <p className="text-sm text-muted-foreground mb-3">
                    {milestone.description}
                  </p>
                  
                  {!milestone.completed && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Progress</span>
                        <span className="font-medium">
                          {milestone.current}/{milestone.target}
                        </span>
                      </div>
                      <Progress 
                        value={(milestone.current / milestone.target) * 100} 
                        className="h-2"
                      />
                    </div>
                  )}
                  
                  {milestone.completed && milestone.reward && (
                    <div className="flex items-center gap-2 text-sm text-success">
                      <Gift className="w-4 h-4" />
                      {milestone.reward}
                    </div>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </TabsContent>

        {/* Progress Tab */}
        <TabsContent value="progress" className="space-y-4 mt-4">
          {/* Resilience Growth Tree */}
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-green-500" />
              Resilience Growth Tree
            </h3>
            
            <div className="flex justify-center">
              <div className="relative">
                {/* Tree visualization */}
                <div className="text-center">
                  <div className="text-7xl mb-2">🌳</div>
                  <div className="text-lg font-semibold">Flourishing</div>
                  <div className="text-sm text-muted-foreground">Level 4 Growth</div>
                </div>
                
                <div className="mt-4 grid grid-cols-4 gap-2 text-center text-xs">
                  <div>
                    <div className="text-2xl mb-1">🌱</div>
                    <div className="text-muted-foreground">Seed</div>
                  </div>
                  <div>
                    <div className="text-2xl mb-1">🌿</div>
                    <div className="text-muted-foreground">Sprout</div>
                  </div>
                  <div>
                    <div className="text-2xl mb-1">🌲</div>
                    <div className="text-muted-foreground">Sapling</div>
                  </div>
                  <div>
                    <div className="text-2xl mb-1 opacity-50">🌳</div>
                    <div className="text-muted-foreground">Tree</div>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Emotional Range */}
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Heart className="w-5 h-5 text-pink-500" />
              Emotional Range Expansion
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span>Emotions Identified</span>
                <span className="font-semibold">24 unique</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {['happy', 'calm', 'grateful', 'hopeful', 'content', 'anxious', 'stressed', 'tired'].map((emotion) => (
                  <Badge key={emotion} variant="outline" className="capitalize">
                    {emotion}
                  </Badge>
                ))}
                <Badge variant="secondary">+16 more</Badge>
              </div>
            </div>
          </Card>

          {/* Coping Skills */}
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-primary" />
              Coping Skill Inventory
            </h3>
            
            <div className="space-y-3">
              {[
                { skill: 'Deep Breathing', uses: 47, mastery: 85 },
                { skill: 'Gratitude Practice', uses: 32, mastery: 70 },
                { skill: 'Mindful Walking', uses: 18, mastery: 55 },
                { skill: 'Journaling', uses: 28, mastery: 65 },
              ].map((item) => (
                <div key={item.skill} className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="flex justify-between text-sm mb-1">
                      <span>{item.skill}</span>
                      <span className="text-muted-foreground">{item.uses} uses</span>
                    </div>
                    <Progress value={item.mastery} className="h-2" />
                  </div>
                  <Badge variant={item.mastery >= 80 ? 'default' : 'outline'}>
                    {item.mastery}%
                  </Badge>
                </div>
              ))}
            </div>
          </Card>

          {/* Share Progress */}
          <Card className="p-4 bg-primary/5 border-primary/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Share2 className="w-5 h-5 text-primary" />
                <div>
                  <div className="font-medium">Share Your Progress</div>
                  <div className="text-sm text-muted-foreground">
                    Celebrate milestones with the community
                  </div>
                </div>
              </div>
              <Button variant="outline" size="sm">
                Share
              </Button>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default WellnessJourney;
