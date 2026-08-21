import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import MockDataBanner from '@/components/MockDataBanner';
import { 
  Sun, 
  Moon, 
  Sunrise, 
  Sunset,
  Frown,
  Smile,
  Battery,
  BatteryFull,
  Leaf,
  CloudLightning,
  CheckCircle,
  Clock,
  Sparkles,
  Send
} from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import AetherMindAPIService from '@/services/AetherMindAPI';

interface CheckInData {
  mood: number;
  energy: number;
  stress: number;
  journal: string;
  timeOfDay: 'morning' | 'afternoon' | 'evening' | 'night';
  timestamp: Date;
}

interface DailyCheckInProps {
  onComplete?: (data: CheckInData) => void;
}

const DailyCheckIn: React.FC<DailyCheckInProps> = ({ onComplete }) => {
  const [mood, setMood] = useState<number[]>([5]);
  const [energy, setEnergy] = useState<number[]>([5]);
  const [stress, setStress] = useState<number[]>([5]);
  const [journal, setJournal] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [todayCheckins, setTodayCheckins] = useState<CheckInData[]>([]);
  const [showQuickTips, setShowQuickTips] = useState(false);

  const getTimeOfDay = (): 'morning' | 'afternoon' | 'evening' | 'night' => {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 17) return 'afternoon';
    if (hour >= 17 && hour < 21) return 'evening';
    return 'night';
  };

  const getTimeIcon = () => {
    const timeOfDay = getTimeOfDay();
    switch (timeOfDay) {
      case 'morning': return <Sunrise className="w-5 h-5 text-amber-500" />;
      case 'afternoon': return <Sun className="w-5 h-5 text-yellow-500" />;
      case 'evening': return <Sunset className="w-5 h-5 text-orange-500" />;
      case 'night': return <Moon className="w-5 h-5 text-indigo-400" />;
    }
  };

  const getTimeGreeting = () => {
    const timeOfDay = getTimeOfDay();
    switch (timeOfDay) {
      case 'morning': return 'Good morning! How are you starting your day?';
      case 'afternoon': return 'Good afternoon! How is your day going?';
      case 'evening': return 'Good evening! How was your day?';
      case 'night': return 'Late night check-in. How are you feeling?';
    }
  };

  const getMoodEmoji = (value: number) => {
    if (value <= 2) return '😞';
    if (value <= 4) return '😕';
    if (value <= 6) return '😐';
    if (value <= 8) return '🙂';
    return '😊';
  };

  const getEnergyEmoji = (value: number) => {
    if (value <= 2) return '💤';
    if (value <= 4) return '😴';
    if (value <= 6) return '🔋';
    if (value <= 8) return '⚡';
    return '🌟';
  };

  const getStressEmoji = (value: number) => {
    if (value <= 2) return '🍃';
    if (value <= 4) return '🌿';
    if (value <= 6) return '💨';
    if (value <= 8) return '⚡';
    return '🌪️';
  };

  const getQuickTip = () => {
    const tips = [];
    if (mood[0] <= 4) {
      tips.push("Consider a 5-minute gratitude exercise");
    }
    if (energy[0] <= 4) {
      tips.push("A short walk or stretch might help boost your energy");
    }
    if (stress[0] >= 7) {
      tips.push("Try the 4-7-8 breathing exercise to reduce stress");
    }
    if (tips.length === 0) {
      tips.push("You're doing great! Keep up the positive momentum");
    }
    return tips;
  };

  useEffect(() => {
    // Load today's check-ins
    const loadTodayCheckins = async () => {
      try {
        const checkins = await AetherMindAPIService.getTodayCheckins();
        setTodayCheckins(checkins);
      } catch (error) {
        console.error('Failed to load check-ins:', error);
      }
    };
    loadTodayCheckins();
  }, []);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    
    try {
      const checkInData: CheckInData = {
        mood: mood[0],
        energy: energy[0],
        stress: stress[0],
        journal: journal.trim(),
        timeOfDay: getTimeOfDay(),
        timestamp: new Date()
      };

      await AetherMindAPIService.saveCheckIn(checkInData);
      
      setTodayCheckins([...todayCheckins, checkInData]);
      
      toast({
        title: "Check-in saved! ✨",
        description: "Your wellness data has been recorded.",
      });

      // Reset form
      setMood([5]);
      setEnergy([5]);
      setStress([5]);
      setJournal('');
      setShowQuickTips(true);
      
      onComplete?.(checkInData);
    } catch (error) {
      toast({
        title: "Error saving check-in",
        description: "Please try again",
        variant: "destructive"
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <MockDataBanner feature="Daily Check-in" storageType="localStorage" />
      
      {/* Header */}
      <Card className="p-6 bg-gradient-to-br from-primary/10 via-secondary/5 to-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            {getTimeIcon()}
            <div>
              <h2 className="text-xl font-semibold">Daily Check-in</h2>
              <p className="text-sm text-muted-foreground">{getTimeGreeting()}</p>
            </div>
          </div>
          <Badge variant="outline" className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            ~30 sec
          </Badge>
        </div>

        {/* Today's Check-ins */}
        {todayCheckins.length > 0 && (
          <div className="flex items-center gap-2 mb-4 p-3 bg-success/10 rounded-lg border border-success/20">
            <CheckCircle className="w-4 h-4 text-success" />
            <span className="text-sm">
              {todayCheckins.length} check-in{todayCheckins.length > 1 ? 's' : ''} today
            </span>
            <div className="flex gap-1 ml-auto">
              {todayCheckins.map((c, i) => (
                <Badge key={i} variant="secondary" className="text-xs">
                  {c.timeOfDay === 'morning' ? 'AM' : 
                   c.timeOfDay === 'afternoon' ? 'PM' : 
                   c.timeOfDay === 'evening' ? 'Eve' : 'Night'}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </Card>

      {/* Sliders */}
      <Card className="p-6 space-y-8">
        {/* Mood Slider */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{getMoodEmoji(mood[0])}</span>
              <div>
                <h3 className="font-medium">Mood</h3>
                <p className="text-xs text-muted-foreground">How are you feeling emotionally?</p>
              </div>
            </div>
            <Badge variant="outline" className="text-lg px-3">
              {mood[0]}/10
            </Badge>
          </div>
          <div className="flex items-center gap-4">
            <Frown className="w-5 h-5 text-muted-foreground" />
            <Slider
              value={mood}
              onValueChange={setMood}
              max={10}
              min={1}
              step={1}
              className="flex-1"
            />
            <Smile className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground px-2">
            <span>Low</span>
            <span>Neutral</span>
            <span>Great</span>
          </div>
        </div>

        {/* Energy Slider */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{getEnergyEmoji(energy[0])}</span>
              <div>
                <h3 className="font-medium">Energy</h3>
                <p className="text-xs text-muted-foreground">How energized do you feel?</p>
              </div>
            </div>
            <Badge variant="outline" className="text-lg px-3">
              {energy[0]}/10
            </Badge>
          </div>
          <div className="flex items-center gap-4">
            <Battery className="w-5 h-5 text-muted-foreground" />
            <Slider
              value={energy}
              onValueChange={setEnergy}
              max={10}
              min={1}
              step={1}
              className="flex-1"
            />
            <BatteryFull className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground px-2">
            <span>Exhausted</span>
            <span>Moderate</span>
            <span>Energized</span>
          </div>
        </div>

        {/* Stress Slider */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{getStressEmoji(stress[0])}</span>
              <div>
                <h3 className="font-medium">Stress</h3>
                <p className="text-xs text-muted-foreground">What's your stress level?</p>
              </div>
            </div>
            <Badge variant="outline" className="text-lg px-3">
              {stress[0]}/10
            </Badge>
          </div>
          <div className="flex items-center gap-4">
            <Leaf className="w-5 h-5 text-muted-foreground" />
            <Slider
              value={stress}
              onValueChange={setStress}
              max={10}
              min={1}
              step={1}
              className="flex-1"
            />
            <CloudLightning className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground px-2">
            <span>Calm</span>
            <span>Moderate</span>
            <span>Overwhelmed</span>
          </div>
        </div>
      </Card>

      {/* Optional Journal */}
      <Card className="p-6">
        <h3 className="font-medium mb-3 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          One-sentence Journal (Optional)
        </h3>
        <Textarea
          placeholder="Today I feel..."
          value={journal}
          onChange={(e) => setJournal(e.target.value)}
          className="min-h-[80px] resize-none"
          maxLength={280}
        />
        <div className="flex justify-between items-center mt-2">
          <span className="text-xs text-muted-foreground">
            {journal.length}/280 characters
          </span>
        </div>
      </Card>

      {/* Quick Tips (shown after submission) */}
      {showQuickTips && (
        <Card className="p-6 bg-primary/5 border-primary/20 animate-fade-in-up">
          <h3 className="font-medium mb-3 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary" />
            Quick Tips Based on Your Check-in
          </h3>
          <ul className="space-y-2">
            {getQuickTip().map((tip, index) => (
              <li key={index} className="flex items-start gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-success mt-0.5 flex-shrink-0" />
                {tip}
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Submit Button */}
      <Button 
        onClick={handleSubmit} 
        disabled={isSubmitting}
        className="w-full h-12 text-lg"
        size="lg"
      >
        {isSubmitting ? (
          <>Saving...</>
        ) : (
          <>
            <Send className="w-5 h-5 mr-2" />
            Complete Check-in
          </>
        )}
      </Button>
    </div>
  );
};

export default DailyCheckIn;
