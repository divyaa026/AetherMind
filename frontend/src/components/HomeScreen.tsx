import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Sun,
  Moon,
  Sunrise,
  Sunset,
  Heart,
  Wind,
  Shield,
  TrendingUp,
  Quote,
  Calendar,
  Target
} from 'lucide-react';

interface HomeScreenProps {
  onNavigate: (tab: string) => void;
}

const HomeScreen: React.FC<HomeScreenProps> = ({ onNavigate }) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [greeting, setGreeting] = useState('');

  const inspirationalQuotes = [
    "You are stronger than you think, braver than you feel, and more loved than you know.",
    "Every small step forward is progress. Be gentle with yourself today.",
    "Your current situation is not your final destination. Keep going.",
    "Healing isn't linear. Allow yourself to feel and grow at your own pace.",
    "You've survived 100% of your difficult days. You're resilient.",
    "It's okay to not be okay. It's okay to ask for help. It's okay to take your time."
  ];

  const [todaysQuote] = useState(
    inspirationalQuotes[Math.floor(Math.random() * inspirationalQuotes.length)]
  );

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const hour = currentTime.getHours();
    if (hour < 12) setGreeting('Good Morning');
    else if (hour < 17) setGreeting('Good Afternoon');
    else setGreeting('Good Evening');
  }, [currentTime]);

  const getTimeIcon = () => {
    const hour = currentTime.getHours();
    if (hour < 6) return Moon;
    if (hour < 12) return Sunrise;
    if (hour < 17) return Sun;
    if (hour < 20) return Sunset;
    return Moon;
  };

  const TimeIcon = getTimeIcon();

  const quickActions = [
    {
      id: 'breathing',
      title: 'Quick Breathing',
      subtitle: '2-minute calm',
      icon: Wind,
      color: 'bg-gradient-breath',
      textColor: 'text-primary-foreground'
    },
    {
      id: 'journal',
      title: 'Express Feelings',
      subtitle: 'Emotion journal',
      icon: Heart,
      color: 'bg-secondary/20',
      textColor: 'text-secondary-foreground'
    },
    {
      id: 'safety',
      title: 'Safety Check',
      subtitle: 'Crisis support',
      icon: Shield,
      color: 'bg-primary/10',
      textColor: 'text-primary-foreground'
    },
    {
      id: 'growth',
      title: 'View Progress',
      subtitle: 'Growth insights',
      icon: TrendingUp,
      color: 'bg-success/10',
      textColor: 'text-success-foreground'
    }
  ];

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* Welcome Card */}
      <Card className="p-6 bg-gradient-calm shadow-embrace">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <TimeIcon className="w-8 h-8 text-primary animate-glow" />
            <div>
              <h2 className="text-2xl font-semibold text-foreground">
                {greeting}
              </h2>
              <p className="text-muted-foreground">
                {currentTime.toLocaleDateString('en-US', { 
                  weekday: 'long', 
                  month: 'long', 
                  day: 'numeric' 
                })}
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm text-muted-foreground">Current time</div>
            <div className="text-lg font-mono">
              {currentTime.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit'
              })}
            </div>
          </div>
        </div>
        
        <div className="bg-primary/5 rounded-lg p-4 border border-primary/10">
          <div className="flex items-start gap-3">
            <Quote className="w-5 h-5 text-primary mt-1 flex-shrink-0" />
            <p className="text-foreground italic leading-relaxed">
              {todaysQuote}
            </p>
          </div>
        </div>
      </Card>

      {/* Quick Actions */}
      <div>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-primary" />
          Quick Actions
        </h3>
        
        <div className="grid grid-cols-2 gap-4">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <Button
                key={action.id}
                variant="ghost"
                onClick={() => onNavigate(action.id)}
                className={`
                  ${action.color} ${action.textColor}
                  h-auto p-4 flex flex-col items-center gap-3
                  hover:scale-105 transition-all duration-300
                  hover:shadow-embrace group
                `}
              >
                <Icon className="w-8 h-8 group-hover:animate-pulse-gentle" />
                <div className="text-center">
                  <div className="font-medium">{action.title}</div>
                  <div className="text-xs opacity-80">{action.subtitle}</div>
                </div>
              </Button>
            );
          })}
        </div>
      </div>

      {/* Daily Check-in */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Calendar className="w-5 h-5 text-secondary" />
            Today's Check-in
          </h3>
          <Badge variant="outline" className="animate-pulse-gentle">
            Daily goal
          </Badge>
        </div>
        
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-success rounded-full"></div>
              <span className="text-sm">Morning breathing</span>
            </div>
            <Badge className="bg-success text-success-foreground">Done</Badge>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-muted-foreground rounded-full"></div>
              <span className="text-sm">Emotion journal</span>
            </div>
            <Button 
              size="sm" 
              variant="outline"
              onClick={() => onNavigate('journal')}
            >
              Start
            </Button>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-muted-foreground rounded-full"></div>
              <span className="text-sm">Evening reflection</span>
            </div>
            <Badge variant="secondary">Later</Badge>
          </div>
        </div>
      </Card>

      {/* Emergency Support */}
      <Card className="p-4 bg-emergency/5 border-emergency/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-5 h-5 text-emergency" />
            <div>
              <div className="font-medium text-emergency">Need immediate support?</div>
              <div className="text-xs text-muted-foreground">24/7 crisis resources available</div>
            </div>
          </div>
          <Button 
            size="sm" 
            className="bg-emergency hover:bg-emergency/90 text-emergency-foreground"
            onClick={() => onNavigate('safety')}
          >
            Get Help
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default HomeScreen;