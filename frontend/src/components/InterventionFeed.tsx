import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import MockDataBanner from '@/components/MockDataBanner';
import {
  Wind,
  Heart,
  Smartphone,
  Users,
  Trees,
  Pencil,
  Clock,
  Sparkles,
  ChevronRight,
  Check,
  X,
  MapPin,
  Moon,
  Sun,
  Coffee,
  Target,
  Trophy,
  Flame,
  RefreshCw
} from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import AetherMindAPIService, { Intervention, WeeklyChallenge, ContextPrompt } from '@/services/AetherMindAPI';

interface InterventionFeedProps {
  onNavigate?: (tab: string) => void;
  userContext?: {
    location?: string;
    timeOfDay?: string;
    currentActivity?: string;
  };
}

const InterventionFeed: React.FC<InterventionFeedProps> = ({ onNavigate, userContext }) => {
  const [interventions, setInterventions] = useState<Intervention[]>([]);
  const [weeklyChallenges, setWeeklyChallenges] = useState<WeeklyChallenge[]>([]);
  const [contextPrompts, setContextPrompts] = useState<ContextPrompt[]>([]);
  const [completedToday, setCompletedToday] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('micro');

  useEffect(() => {
    loadInterventions();
  }, []);

  const loadInterventions = async () => {
    setLoading(true);
    try {
      const [interventionsData, challengesData, promptsData] = await Promise.all([
        AetherMindAPIService.getMicroInterventions(),
        AetherMindAPIService.getWeeklyChallenges(),
        AetherMindAPIService.getContextPrompts()
      ]);
      setInterventions(interventionsData);
      setWeeklyChallenges(challengesData);
      setContextPrompts(promptsData);
    } catch (error) {
      console.error('Failed to load interventions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCompleteIntervention = async (id: string) => {
    try {
      await AetherMindAPIService.completeIntervention(id);
      setCompletedToday([...completedToday, id]);
      toast({
        title: "Great job! 🎉",
        description: "You've completed this wellness activity.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to record completion",
        variant: "destructive"
      });
    }
  };

  const handleDismissPrompt = (id: string) => {
    setContextPrompts(contextPrompts.filter(p => p.id !== id));
  };

  const getInterventionIcon = (type: string) => {
    switch (type) {
      case 'breathing': return <Wind className="w-5 h-5" />;
      case 'gratitude': return <Pencil className="w-5 h-5" />;
      case 'mindfulness': return <Sparkles className="w-5 h-5" />;
      case 'social': return <Users className="w-5 h-5" />;
      case 'nature': return <Trees className="w-5 h-5" />;
      case 'digital-detox': return <Smartphone className="w-5 h-5" />;
      default: return <Heart className="w-5 h-5" />;
    }
  };

  const getChallengeIcon = (type: string) => {
    switch (type) {
      case 'digital-detox': return <Smartphone className="w-6 h-6" />;
      case 'social': return <Users className="w-6 h-6" />;
      case 'sleep': return <Moon className="w-6 h-6" />;
      default: return <Target className="w-6 h-6" />;
    }
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-muted rounded w-1/3"></div>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 bg-muted rounded"></div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <MockDataBanner feature="Interventions Feed" storageType="demo" />
      
      {/* Header */}
      <Card className="p-6 bg-gradient-to-br from-secondary/10 via-primary/5 to-background">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <Sparkles className="w-6 h-6 text-primary" />
            <div>
              <h2 className="text-xl font-semibold">Personalized Interventions</h2>
              <p className="text-sm text-muted-foreground">
                Activities tailored to your wellness patterns
              </p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={loadInterventions}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>

        <div className="flex items-center gap-2 mt-4">
          <Badge variant="secondary" className="flex items-center gap-1">
            <Flame className="w-3 h-3 text-orange-500" />
            {completedToday.length} completed today
          </Badge>
        </div>
      </Card>

      {/* Context-Aware Prompts */}
      {contextPrompts.length > 0 && (
        <Card className="p-4 border-primary/20 bg-primary/5">
          <div className="space-y-3">
            {contextPrompts.map((prompt) => (
              <div key={prompt.id} className="flex items-center justify-between p-3 bg-background rounded-lg border">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                    {prompt.context === 'work' && <Coffee className="w-5 h-5 text-primary" />}
                    {prompt.context === 'late-night' && <Moon className="w-5 h-5 text-primary" />}
                    {prompt.context === 'weekend' && <Sun className="w-5 h-5 text-primary" />}
                    {prompt.context === 'location' && <MapPin className="w-5 h-5 text-primary" />}
                  </div>
                  <div>
                    <div className="font-medium text-sm">{prompt.title}</div>
                    <div className="text-xs text-muted-foreground">{prompt.suggestion}</div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button 
                    size="sm" 
                    variant="ghost"
                    onClick={() => handleDismissPrompt(prompt.id)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                  <Button 
                    size="sm"
                    onClick={() => {
                      handleCompleteIntervention(prompt.id);
                      handleDismissPrompt(prompt.id);
                    }}
                  >
                    <Check className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="micro">Micro-Interventions</TabsTrigger>
          <TabsTrigger value="challenges">Weekly Challenges</TabsTrigger>
        </TabsList>

        {/* Micro-Interventions */}
        <TabsContent value="micro" className="space-y-4 mt-4">
          {interventions.map((intervention) => (
            <Card 
              key={intervention.id}
              className={`p-4 transition-all duration-300 ${
                completedToday.includes(intervention.id) 
                  ? 'bg-success/5 border-success/20' 
                  : 'hover:border-primary/30'
              }`}
            >
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                  completedToday.includes(intervention.id)
                    ? 'bg-success/20 text-success'
                    : 'bg-primary/10 text-primary'
                }`}>
                  {completedToday.includes(intervention.id) 
                    ? <Check className="w-6 h-6" />
                    : getInterventionIcon(intervention.type)
                  }
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-medium">{intervention.title}</h3>
                    <Badge variant="outline" className="text-xs">
                      <Clock className="w-3 h-3 mr-1" />
                      {intervention.duration}
                    </Badge>
                  </div>
                  
                  <p className="text-sm text-muted-foreground mb-3">
                    {intervention.description}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex gap-2">
                      {intervention.tags.map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    
                    {!completedToday.includes(intervention.id) && (
                      <Button 
                        size="sm"
                        onClick={() => {
                          if (intervention.navigateTo) {
                            onNavigate?.(intervention.navigateTo);
                          } else {
                            handleCompleteIntervention(intervention.id);
                          }
                        }}
                      >
                        {intervention.navigateTo ? 'Start' : 'Done'}
                        <ChevronRight className="w-4 h-4 ml-1" />
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </TabsContent>

        {/* Weekly Challenges */}
        <TabsContent value="challenges" className="space-y-4 mt-4">
          {weeklyChallenges.map((challenge) => (
            <Card key={challenge.id} className="p-6">
              <div className="flex items-start gap-4">
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${
                  challenge.completed 
                    ? 'bg-success/20 text-success' 
                    : 'bg-gradient-to-br from-primary/20 to-secondary/20 text-primary'
                }`}>
                  {challenge.completed 
                    ? <Trophy className="w-7 h-7" />
                    : getChallengeIcon(challenge.type)
                  }
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold text-lg">{challenge.title}</h3>
                    {challenge.completed && (
                      <Badge className="bg-success text-success-foreground">
                        Completed!
                      </Badge>
                    )}
                  </div>
                  
                  <p className="text-sm text-muted-foreground mb-4">
                    {challenge.description}
                  </p>
                  
                  {/* Progress */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Progress</span>
                      <span className="font-medium">
                        {challenge.currentProgress}/{challenge.targetProgress} {challenge.unit}
                      </span>
                    </div>
                    <Progress 
                      value={(challenge.currentProgress / challenge.targetProgress) * 100} 
                      className="h-2"
                    />
                  </div>
                  
                  {/* Rewards */}
                  <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                    <div className="flex items-center gap-2 text-sm">
                      <Sparkles className="w-4 h-4 text-amber-500" />
                      <span className="font-medium">Reward:</span>
                      <span className="text-muted-foreground">{challenge.reward}</span>
                    </div>
                  </div>
                  
                  {/* Days remaining */}
                  <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
                    <Clock className="w-4 h-4" />
                    {challenge.daysRemaining} days remaining
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default InterventionFeed;
