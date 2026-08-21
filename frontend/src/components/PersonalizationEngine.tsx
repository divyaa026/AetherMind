import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import MockDataBanner from '@/components/MockDataBanner';
import {
  Settings,
  Sliders,
  Target,
  Clock,
  Bell,
  BookOpen,
  Video,
  Pencil,
  Users,
  Moon,
  Sun,
  Brain,
  Heart,
  Zap,
  Coffee,
  CheckCircle,
  Plus,
  X,
  ChevronRight,
  Sparkles,
  Save
} from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import AetherMindAPIService, { UserPreferences, WellnessGoal } from '@/services/AetherMindAPI';

interface PersonalizationEngineProps {
  onNavigate?: (tab: string) => void;
}

const PersonalizationEngine: React.FC<PersonalizationEngineProps> = ({ onNavigate }) => {
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);
  const [goals, setGoals] = useState<WellnessGoal[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('interventions');

  const predefinedGoals = [
    { id: 'stress', label: 'Reduce work stress', icon: <Coffee className="w-4 h-4" /> },
    { id: 'energy', label: 'Improve morning energy', icon: <Sun className="w-4 h-4" /> },
    { id: 'social', label: 'Build social connections', icon: <Users className="w-4 h-4" /> },
    { id: 'sleep', label: 'Better sleep quality', icon: <Moon className="w-4 h-4" /> },
    { id: 'mindfulness', label: 'Daily mindfulness practice', icon: <Brain className="w-4 h-4" /> },
    { id: 'anxiety', label: 'Manage anxiety', icon: <Heart className="w-4 h-4" /> },
  ];

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    setLoading(true);
    try {
      const [prefsData, goalsData] = await Promise.all([
        AetherMindAPIService.getUserPreferences(),
        AetherMindAPIService.getWellnessGoals()
      ]);
      setPreferences(prefsData);
      setGoals(goalsData);
    } catch (error) {
      console.error('Failed to load preferences:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSavePreferences = async () => {
    if (!preferences) return;
    
    setSaving(true);
    try {
      await AetherMindAPIService.saveUserPreferences(preferences);
      toast({
        title: "Preferences saved!",
        description: "Your personalization settings have been updated.",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save preferences",
        variant: "destructive"
      });
    } finally {
      setSaving(false);
    }
  };

  const handleToggleGoal = (goalId: string) => {
    const existingGoal = goals.find(g => g.id === goalId);
    if (existingGoal) {
      setGoals(goals.filter(g => g.id !== goalId));
    } else {
      const goalDef = predefinedGoals.find(g => g.id === goalId);
      if (goalDef) {
        setGoals([...goals, { id: goalId, label: goalDef.label, active: true }]);
      }
    }
  };

  const updatePreference = (key: string, value: any) => {
    if (!preferences) return;
    setPreferences({ ...preferences, [key]: value });
  };

  if (loading || !preferences) {
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
      <MockDataBanner feature="Personalization Engine" storageType="localStorage" />
      
      {/* Header */}
      <Card className="p-6 bg-gradient-to-br from-primary/10 via-secondary/5 to-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Sliders className="w-6 h-6 text-primary" />
            <div>
              <h2 className="text-xl font-semibold">Personalization Engine</h2>
              <p className="text-sm text-muted-foreground">
                Customize your wellness experience
              </p>
            </div>
          </div>
          <Button onClick={handleSavePreferences} disabled={saving}>
            {saving ? (
              <>Saving...</>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save All
              </>
            )}
          </Button>
        </div>
      </Card>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="interventions">
            <Sparkles className="w-4 h-4 mr-2" />
            Interventions
          </TabsTrigger>
          <TabsTrigger value="learning">
            <BookOpen className="w-4 h-4 mr-2" />
            Learning
          </TabsTrigger>
          <TabsTrigger value="goals">
            <Target className="w-4 h-4 mr-2" />
            Goals
          </TabsTrigger>
        </TabsList>

        {/* Intervention Preferences */}
        <TabsContent value="interventions" className="space-y-4 mt-4">
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5 text-primary" />
              Exercise Duration Preference
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Preferred exercise length
                </span>
                <Badge variant="outline">
                  {preferences.exerciseDuration} min max
                </Badge>
              </div>
              <Slider
                value={[preferences.exerciseDuration]}
                onValueChange={([value]) => updatePreference('exerciseDuration', value)}
                min={1}
                max={15}
                step={1}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Quick (1 min)</span>
                <span>Moderate (5 min)</span>
                <span>Deep (15 min)</span>
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Users className="w-5 h-5 text-secondary" />
              Activity Preferences
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div>
                  <div className="font-medium text-sm">Suggest social activities</div>
                  <div className="text-xs text-muted-foreground">
                    Include friend calls, social connections
                  </div>
                </div>
                <Switch
                  checked={preferences.suggestSocialActivities}
                  onCheckedChange={(checked) => updatePreference('suggestSocialActivities', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div>
                  <div className="font-medium text-sm">Include outdoor activities</div>
                  <div className="text-xs text-muted-foreground">
                    Nature walks, outdoor exercises
                  </div>
                </div>
                <Switch
                  checked={preferences.suggestOutdoorActivities}
                  onCheckedChange={(checked) => updatePreference('suggestOutdoorActivities', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div>
                  <div className="font-medium text-sm">Digital detox suggestions</div>
                  <div className="text-xs text-muted-foreground">
                    Screen break reminders
                  </div>
                </div>
                <Switch
                  checked={preferences.suggestDigitalDetox}
                  onCheckedChange={(checked) => updatePreference('suggestDigitalDetox', checked)}
                />
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-amber-500" />
              Focus Areas
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              Prioritize interventions for specific areas
            </p>
            
            <div className="grid grid-cols-2 gap-3">
              {['Sleep', 'Stress', 'Energy', 'Focus', 'Anxiety', 'Mood'].map((area) => (
                <Button
                  key={area}
                  variant={preferences.focusAreas?.includes(area.toLowerCase()) ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    const current = preferences.focusAreas || [];
                    const updated = current.includes(area.toLowerCase())
                      ? current.filter(a => a !== area.toLowerCase())
                      : [...current, area.toLowerCase()];
                    updatePreference('focusAreas', updated);
                  }}
                >
                  {preferences.focusAreas?.includes(area.toLowerCase()) && (
                    <CheckCircle className="w-4 h-4 mr-2" />
                  )}
                  {area}
                </Button>
              ))}
            </div>
          </Card>
        </TabsContent>

        {/* Learning Style */}
        <TabsContent value="learning" className="space-y-4 mt-4">
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" />
              Content Format
            </h3>
            
            <RadioGroup
              value={preferences.contentFormat}
              onValueChange={(value) => updatePreference('contentFormat', value)}
              className="space-y-3"
            >
              <div className="flex items-center space-x-3 p-3 rounded-lg border hover:border-primary/50 cursor-pointer">
                <RadioGroupItem value="video" id="video" />
                <Label htmlFor="video" className="flex items-center gap-2 cursor-pointer flex-1">
                  <Video className="w-5 h-5 text-primary" />
                  <div>
                    <div className="font-medium">Video Lessons</div>
                    <div className="text-xs text-muted-foreground">
                      Watch guided video content
                    </div>
                  </div>
                </Label>
              </div>
              
              <div className="flex items-center space-x-3 p-3 rounded-lg border hover:border-primary/50 cursor-pointer">
                <RadioGroupItem value="reading" id="reading" />
                <Label htmlFor="reading" className="flex items-center gap-2 cursor-pointer flex-1">
                  <Pencil className="w-5 h-5 text-secondary" />
                  <div>
                    <div className="font-medium">Written Exercises</div>
                    <div className="text-xs text-muted-foreground">
                      Read and complete written activities
                    </div>
                  </div>
                </Label>
              </div>
              
              <div className="flex items-center space-x-3 p-3 rounded-lg border hover:border-primary/50 cursor-pointer">
                <RadioGroupItem value="mixed" id="mixed" />
                <Label htmlFor="mixed" className="flex items-center gap-2 cursor-pointer flex-1">
                  <Sparkles className="w-5 h-5 text-amber-500" />
                  <div>
                    <div className="font-medium">Mixed Format</div>
                    <div className="text-xs text-muted-foreground">
                      Combination of video and written content
                    </div>
                  </div>
                </Label>
              </div>
            </RadioGroup>
          </Card>

          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-secondary" />
              Content Depth
            </h3>
            
            <RadioGroup
              value={preferences.contentDepth}
              onValueChange={(value) => updatePreference('contentDepth', value)}
              className="space-y-3"
            >
              <div className="flex items-center space-x-3 p-3 rounded-lg border hover:border-primary/50 cursor-pointer">
                <RadioGroupItem value="practical" id="practical" />
                <Label htmlFor="practical" className="cursor-pointer flex-1">
                  <div className="font-medium">Practical Tips</div>
                  <div className="text-xs text-muted-foreground">
                    Quick, actionable advice without deep theory
                  </div>
                </Label>
              </div>
              
              <div className="flex items-center space-x-3 p-3 rounded-lg border hover:border-primary/50 cursor-pointer">
                <RadioGroupItem value="balanced" id="balanced" />
                <Label htmlFor="balanced" className="cursor-pointer flex-1">
                  <div className="font-medium">Balanced</div>
                  <div className="text-xs text-muted-foreground">
                    Mix of practical advice and some science
                  </div>
                </Label>
              </div>
              
              <div className="flex items-center space-x-3 p-3 rounded-lg border hover:border-primary/50 cursor-pointer">
                <RadioGroupItem value="science" id="science" />
                <Label htmlFor="science" className="cursor-pointer flex-1">
                  <div className="font-medium">Science-Heavy</div>
                  <div className="text-xs text-muted-foreground">
                    Deep dive into research and methodology
                  </div>
                </Label>
              </div>
            </RadioGroup>
          </Card>

          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Bell className="w-5 h-5 text-primary" />
              Reminder Schedule
            </h3>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div>
                  <div className="font-medium text-sm">Morning check-in</div>
                  <div className="text-xs text-muted-foreground">8:00 AM</div>
                </div>
                <Switch
                  checked={preferences.reminders?.morning}
                  onCheckedChange={(checked) => updatePreference('reminders', { ...preferences.reminders, morning: checked })}
                />
              </div>
              
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div>
                  <div className="font-medium text-sm">Midday reflection</div>
                  <div className="text-xs text-muted-foreground">1:00 PM</div>
                </div>
                <Switch
                  checked={preferences.reminders?.midday}
                  onCheckedChange={(checked) => updatePreference('reminders', { ...preferences.reminders, midday: checked })}
                />
              </div>
              
              <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                <div>
                  <div className="font-medium text-sm">Evening wind-down</div>
                  <div className="text-xs text-muted-foreground">7:00 PM</div>
                </div>
                <Switch
                  checked={preferences.reminders?.evening}
                  onCheckedChange={(checked) => updatePreference('reminders', { ...preferences.reminders, evening: checked })}
                />
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Goals */}
        <TabsContent value="goals" className="space-y-4 mt-4">
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-primary" />
              Your Wellness Goals
            </h3>
            <p className="text-sm text-muted-foreground mb-6">
              Select the goals you want to focus on. We'll personalize your experience accordingly.
            </p>

            <div className="space-y-3">
              {predefinedGoals.map((goal) => {
                const isActive = goals.some(g => g.id === goal.id);
                return (
                  <div
                    key={goal.id}
                    onClick={() => handleToggleGoal(goal.id)}
                    className={`flex items-center justify-between p-4 rounded-lg border cursor-pointer transition-all ${
                      isActive 
                        ? 'bg-primary/5 border-primary/30' 
                        : 'hover:border-primary/30'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        isActive ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'
                      }`}>
                        {goal.icon}
                      </div>
                      <span className="font-medium">{goal.label}</span>
                    </div>
                    
                    {isActive ? (
                      <CheckCircle className="w-5 h-5 text-primary" />
                    ) : (
                      <Plus className="w-5 h-5 text-muted-foreground" />
                    )}
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Active Goals Summary */}
          {goals.length > 0 && (
            <Card className="p-6 bg-primary/5 border-primary/20">
              <h4 className="font-medium mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-primary" />
                Your Focus Areas ({goals.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {goals.map((goal) => (
                  <Badge key={goal.id} variant="secondary" className="flex items-center gap-1">
                    {goal.label}
                    <X 
                      className="w-3 h-3 cursor-pointer hover:text-destructive"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleGoal(goal.id);
                      }}
                    />
                  </Badge>
                ))}
              </div>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PersonalizationEngine;
