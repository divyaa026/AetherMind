import React from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { 
  Wind, 
  Heart, 
  Shield, 
  TrendingUp,
  Home,
  Activity
} from 'lucide-react';

interface NavigationProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const Navigation: React.FC<NavigationProps> = ({ activeTab, onTabChange }) => {
  const navigationItems = [
    {
      id: 'home',
      label: 'Home',
      icon: Home,
      description: 'Your sanctuary'
    },
    {
      id: 'breathing',
      label: 'Breathe',
      icon: Wind,
      description: 'Mindful breathing'
    },
    {
      id: 'journal',
      label: 'Journal',
      icon: Heart,
      description: 'Emotion tracking'
    },
    {
      id: 'safety',
      label: 'Safety',
      icon: Shield,
      description: 'Crisis support'
    },
    {
      id: 'growth',
      label: 'Growth',
      icon: TrendingUp,
      description: 'Progress insights'
    }
  ];

  return (
    <Card className="p-4 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-breath rounded-full flex items-center justify-center animate-pulse-gentle">
            <Activity className="w-5 h-5 text-primary-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-breath bg-clip-text text-transparent">
              AetherMind
            </h1>
            <p className="text-xs text-muted-foreground">Mental wellness companion</p>
          </div>
        </div>
        
        <div className="text-xs text-muted-foreground">
          {new Date().toLocaleDateString('en-US', { 
            weekday: 'long', 
            month: 'short', 
            day: 'numeric' 
          })}
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="grid grid-cols-5 gap-2">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          
          return (
            <Button
              key={item.id}
              variant={isActive ? "default" : "ghost"}
              size="sm"
              onClick={() => onTabChange(item.id)}
              className={`
                flex flex-col items-center gap-1 h-auto py-3 px-2
                transition-all duration-300 group
                ${isActive 
                  ? 'bg-primary text-primary-foreground shadow-gentle animate-fade-in-up' 
                  : 'hover:bg-accent hover:shadow-embrace'
                }
              `}
            >
              <Icon className={`w-5 h-5 transition-transform duration-200 ${
                isActive ? 'scale-110' : 'group-hover:scale-105'
              }`} />
              <span className="text-xs font-medium">{item.label}</span>
            </Button>
          );
        })}
      </div>
    </Card>
  );
};

export default Navigation;