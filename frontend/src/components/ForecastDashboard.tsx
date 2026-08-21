import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import MockDataBanner from '@/components/MockDataBanner';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Clock,
  Sun,
  Moon,
  AlertTriangle,
  Lightbulb,
  BarChart3,
  Activity,
  Zap,
  Brain,
  Target,
  ChevronRight,
  Sparkles
} from 'lucide-react';
import AetherMindAPIService, { EmotionalForecast, PatternInsight, WeeklyHeatmap } from '@/services/AetherMindAPI';

interface ForecastDashboardProps {
  onNavigate?: (tab: string) => void;
}

const ForecastDashboard: React.FC<ForecastDashboardProps> = ({ onNavigate }) => {
  const [timeRange, setTimeRange] = useState<'7' | '30' | '90'>('7');
  const [forecast, setForecast] = useState<EmotionalForecast | null>(null);
  const [patterns, setPatterns] = useState<PatternInsight[]>([]);
  const [heatmapData, setHeatmapData] = useState<WeeklyHeatmap | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const [forecastData, patternsData, heatmap] = await Promise.all([
          AetherMindAPIService.getEmotionalForecast(parseInt(timeRange)),
          AetherMindAPIService.getPatternInsights(),
          AetherMindAPIService.getWeeklyHeatmap()
        ]);
        setForecast(forecastData);
        setPatterns(patternsData);
        setHeatmapData(heatmap);
      } catch (error) {
        console.error('Failed to load forecast data:', error);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [timeRange]);

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-success" />;
      case 'down': return <TrendingDown className="w-4 h-4 text-destructive" />;
      default: return <Minus className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const getHeatmapColor = (value: number) => {
    // Value 0-10, lower stress = green, higher = red
    if (value <= 3) return 'bg-green-400';
    if (value <= 5) return 'bg-green-200';
    if (value <= 6) return 'bg-yellow-200';
    if (value <= 7) return 'bg-orange-300';
    if (value <= 8) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const times = ['Morning', 'Afternoon', 'Evening'];

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
      <MockDataBanner feature="AI Emotional Forecast" storageType="demo" />
      
      {/* Header */}
      <Card className="p-6 bg-gradient-to-br from-primary/10 via-secondary/5 to-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Brain className="w-6 h-6 text-primary" />
            <div>
              <h2 className="text-xl font-semibold">AI Emotional Forecast</h2>
              <p className="text-sm text-muted-foreground">
                Insights powered by your check-in data
              </p>
            </div>
          </div>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2">
          {(['7', '30', '90'] as const).map((range) => (
            <Button
              key={range}
              variant={timeRange === range ? 'default' : 'outline'}
              size="sm"
              onClick={() => setTimeRange(range)}
            >
              {range} days
            </Button>
          ))}
        </div>
      </Card>

      {/* Tomorrow's Forecast */}
      {forecast && (
        <Card className="p-6">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            Tomorrow's Energy Forecast
          </h3>
          
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="text-center p-4 bg-muted/50 rounded-lg">
              <div className="text-3xl mb-2">{forecast.predictedMood >= 7 ? '😊' : forecast.predictedMood >= 4 ? '😐' : '😔'}</div>
              <div className="text-sm text-muted-foreground">Mood</div>
              <div className="font-semibold">{forecast.predictedMood}/10</div>
              <div className="flex items-center justify-center gap-1 mt-1">
                {getTrendIcon(forecast.moodTrend)}
              </div>
            </div>
            
            <div className="text-center p-4 bg-muted/50 rounded-lg">
              <div className="text-3xl mb-2">{forecast.predictedEnergy >= 7 ? '⚡' : forecast.predictedEnergy >= 4 ? '🔋' : '💤'}</div>
              <div className="text-sm text-muted-foreground">Energy</div>
              <div className="font-semibold">{forecast.predictedEnergy}/10</div>
              <div className="flex items-center justify-center gap-1 mt-1">
                {getTrendIcon(forecast.energyTrend)}
              </div>
            </div>
            
            <div className="text-center p-4 bg-muted/50 rounded-lg">
              <div className="text-3xl mb-2">{forecast.predictedStress <= 4 ? '🍃' : forecast.predictedStress <= 7 ? '💨' : '🌪️'}</div>
              <div className="text-sm text-muted-foreground">Stress</div>
              <div className="font-semibold">{forecast.predictedStress}/10</div>
              <div className="flex items-center justify-center gap-1 mt-1">
                {getTrendIcon(forecast.stressTrend === 'up' ? 'down' : forecast.stressTrend === 'down' ? 'up' : 'stable')}
              </div>
            </div>
          </div>

          <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-primary mt-0.5" />
              <div>
                <div className="font-medium mb-1">AI Recommendation</div>
                <p className="text-sm text-muted-foreground">
                  {forecast.recommendation}
                </p>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Weekly Stress Heatmap */}
      <Card className="p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-secondary" />
          Your Weekly Stress Pattern
        </h3>
        
        <div className="overflow-x-auto">
          <div className="min-w-[400px]">
            {/* Header */}
            <div className="grid grid-cols-8 gap-2 mb-2">
              <div className="text-xs text-muted-foreground"></div>
              {days.map((day) => (
                <div key={day} className="text-xs text-center text-muted-foreground font-medium">
                  {day}
                </div>
              ))}
            </div>
            
            {/* Heatmap Grid */}
            {times.map((time, timeIdx) => (
              <div key={time} className="grid grid-cols-8 gap-2 mb-2">
                <div className="text-xs text-muted-foreground flex items-center">
                  {time}
                </div>
                {days.map((day, dayIdx) => {
                  const value = heatmapData?.data[timeIdx]?.[dayIdx] ?? 5;
                  return (
                    <div
                      key={`${time}-${day}`}
                      className={`h-8 rounded ${getHeatmapColor(value)} flex items-center justify-center text-xs font-medium text-white`}
                      title={`${day} ${time}: Stress level ${value}/10`}
                    >
                      {value}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
        
        {/* Legend */}
        <div className="flex items-center justify-center gap-2 mt-4">
          <span className="text-xs text-muted-foreground">Low stress</span>
          <div className="flex gap-1">
            <div className="w-4 h-4 rounded bg-green-400"></div>
            <div className="w-4 h-4 rounded bg-green-200"></div>
            <div className="w-4 h-4 rounded bg-yellow-200"></div>
            <div className="w-4 h-4 rounded bg-orange-300"></div>
            <div className="w-4 h-4 rounded bg-orange-500"></div>
            <div className="w-4 h-4 rounded bg-red-500"></div>
          </div>
          <span className="text-xs text-muted-foreground">High stress</span>
        </div>
      </Card>

      {/* Emotional Timeline */}
      <Card className="p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          Emotional Timeline ({timeRange} days)
        </h3>
        
        <div className="h-48 relative">
          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 bottom-8 w-8 flex flex-col justify-between text-xs text-muted-foreground">
            <span>10</span>
            <span>5</span>
            <span>0</span>
          </div>
          
          {/* Chart Area */}
          <div className="ml-10 h-full flex flex-col">
            <div className="flex-1 relative border-l border-b border-border">
              {/* Grid lines */}
              <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
                <div className="border-b border-dashed border-border/50"></div>
                <div className="border-b border-dashed border-border/50"></div>
              </div>
              
              {/* Data visualization */}
              <div className="absolute inset-0 flex items-end justify-around gap-1 p-2">
                {forecast?.timeline.map((point, idx) => (
                  <div key={idx} className="flex-1 flex flex-col gap-1 items-center">
                    {/* Mood bar */}
                    <div
                      className="w-full max-w-4 bg-primary/80 rounded-t transition-all duration-300"
                      style={{ height: `${point.mood * 10}%` }}
                      title={`Mood: ${point.mood}`}
                    />
                    {/* Energy bar */}
                    <div
                      className="w-full max-w-4 bg-secondary/80 rounded-t transition-all duration-300"
                      style={{ height: `${point.energy * 10}%` }}
                      title={`Energy: ${point.energy}`}
                    />
                  </div>
                ))}
              </div>
            </div>
            
            {/* X-axis */}
            <div className="h-8 flex justify-around text-xs text-muted-foreground pt-2">
              {forecast?.timeline.slice(0, 7).map((point, idx) => (
                <span key={idx}>{new Date(point.date).toLocaleDateString('en-US', { weekday: 'short' })}</span>
              ))}
            </div>
          </div>
        </div>
        
        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-primary/80"></div>
            <span className="text-sm text-muted-foreground">Mood</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-secondary/80"></div>
            <span className="text-sm text-muted-foreground">Energy</span>
          </div>
        </div>
      </Card>

      {/* Pattern Insights */}
      <Card className="p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-amber-500" />
          Patterns Spotted
        </h3>
        
        <div className="space-y-4">
          {patterns.map((pattern, idx) => (
            <div
              key={idx}
              className="p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors cursor-pointer"
              onClick={() => pattern.actionTab && onNavigate?.(pattern.actionTab)}
            >
              <div className="flex items-start gap-3">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  pattern.type === 'positive' ? 'bg-success/20 text-success' :
                  pattern.type === 'negative' ? 'bg-destructive/20 text-destructive' :
                  'bg-primary/20 text-primary'
                }`}>
                  {pattern.icon === 'calendar' && <Calendar className="w-5 h-5" />}
                  {pattern.icon === 'activity' && <Activity className="w-5 h-5" />}
                  {pattern.icon === 'moon' && <Moon className="w-5 h-5" />}
                  {pattern.icon === 'zap' && <Zap className="w-5 h-5" />}
                  {pattern.icon === 'target' && <Target className="w-5 h-5" />}
                </div>
                
                <div className="flex-1">
                  <div className="font-medium mb-1">{pattern.title}</div>
                  <p className="text-sm text-muted-foreground">{pattern.description}</p>
                  
                  {pattern.stat && (
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant={pattern.type === 'positive' ? 'default' : pattern.type === 'negative' ? 'destructive' : 'secondary'}>
                        {pattern.stat}
                      </Badge>
                    </div>
                  )}
                </div>
                
                {pattern.actionTab && (
                  <ChevronRight className="w-5 h-5 text-muted-foreground" />
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default ForecastDashboard;
