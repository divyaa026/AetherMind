import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Clock,
  Sun,
  Moon,
  Activity,
  Heart,
  Zap,
  Brain,
  Target,
  Download,
  Share2,
  Sparkles,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  RefreshCw,
  ChevronRight,
  FileText,
  GitCompare
} from 'lucide-react';
import AetherMindAPIService, { CorrelationData, ProgressReport, CustomGoal } from '@/services/AetherMindAPI';

interface AnalyticsPortalProps {
  onNavigate?: (tab: string) => void;
}

const AnalyticsPortal: React.FC<AnalyticsPortalProps> = ({ onNavigate }) => {
  const [timeRange, setTimeRange] = useState<'7' | '30' | '90' | '365'>('30');
  const [correlations, setCorrelations] = useState<CorrelationData[]>([]);
  const [progressReport, setProgressReport] = useState<ProgressReport | null>(null);
  const [goals, setGoals] = useState<CustomGoal[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('correlations');

  useEffect(() => {
    loadAnalytics();
  }, [timeRange]);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const [correlationsData, reportData, goalsData] = await Promise.all([
        AetherMindAPIService.getCorrelations(parseInt(timeRange)),
        AetherMindAPIService.getProgressReport(parseInt(timeRange)),
        AetherMindAPIService.getCustomGoals()
      ]);
      setCorrelations(correlationsData);
      setProgressReport(reportData);
      setGoals(goalsData);
    } catch (error) {
      console.error('Failed to load analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCorrelationColor = (value: number) => {
    if (value >= 0.6) return 'text-success';
    if (value >= 0.3) return 'text-amber-500';
    if (value >= 0) return 'text-muted-foreground';
    if (value >= -0.3) return 'text-orange-500';
    return 'text-destructive';
  };

  const getCorrelationBg = (value: number) => {
    if (value >= 0.6) return 'bg-success/10';
    if (value >= 0.3) return 'bg-amber-500/10';
    if (value >= 0) return 'bg-muted/50';
    if (value >= -0.3) return 'bg-orange-500/10';
    return 'bg-destructive/10';
  };

  const formatCorrelation = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}`;
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-success" />;
      case 'down': return <TrendingDown className="w-4 h-4 text-destructive" />;
      default: return <Minus className="w-4 h-4 text-muted-foreground" />;
    }
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-muted rounded w-1/3"></div>
          <div className="h-64 bg-muted rounded"></div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6 bg-gradient-to-br from-primary/10 via-secondary/5 to-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-primary" />
            <div>
              <h2 className="text-xl font-semibold">Data Insights Portal</h2>
              <p className="text-sm text-muted-foreground">
                Discover patterns and track your wellness journey
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Select value={timeRange} onValueChange={(v) => setTimeRange(v as typeof timeRange)}>
              <SelectTrigger className="w-[120px]">
                <SelectValue placeholder="Time range" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="7">7 days</SelectItem>
                <SelectItem value="30">30 days</SelectItem>
                <SelectItem value="90">90 days</SelectItem>
                <SelectItem value="365">1 year</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="icon">
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Quick Stats */}
        {progressReport && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-background/50 rounded-lg border">
              <div className="text-xs text-muted-foreground mb-1">Avg Mood</div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold">{progressReport.avgMood.toFixed(1)}</span>
                {getTrendIcon(progressReport.moodTrend)}
              </div>
            </div>
            <div className="p-3 bg-background/50 rounded-lg border">
              <div className="text-xs text-muted-foreground mb-1">Avg Energy</div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold">{progressReport.avgEnergy.toFixed(1)}</span>
                {getTrendIcon(progressReport.energyTrend)}
              </div>
            </div>
            <div className="p-3 bg-background/50 rounded-lg border">
              <div className="text-xs text-muted-foreground mb-1">Avg Stress</div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold">{progressReport.avgStress.toFixed(1)}</span>
                {getTrendIcon(progressReport.stressTrend === 'down' ? 'up' : progressReport.stressTrend === 'up' ? 'down' : 'stable')}
              </div>
            </div>
            <div className="p-3 bg-background/50 rounded-lg border">
              <div className="text-xs text-muted-foreground mb-1">Check-ins</div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold">{progressReport.totalCheckins}</span>
                <Badge variant="outline" className="text-xs">
                  {Math.round(progressReport.totalCheckins / parseInt(timeRange) * 100)}% rate
                </Badge>
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="correlations">
            <GitCompare className="w-4 h-4 mr-2" />
            Correlations
          </TabsTrigger>
          <TabsTrigger value="patterns">
            <Activity className="w-4 h-4 mr-2" />
            Patterns
          </TabsTrigger>
          <TabsTrigger value="goals">
            <Target className="w-4 h-4 mr-2" />
            Goals
          </TabsTrigger>
        </TabsList>

        {/* Correlations Tab */}
        <TabsContent value="correlations" className="space-y-4 mt-4">
          <Card className="p-6">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <GitCompare className="w-5 h-5 text-primary" />
              Correlation Explorer
            </h3>
            <p className="text-sm text-muted-foreground mb-6">
              Discover how different factors affect your wellness metrics.
            </p>

            <div className="space-y-4">
              {correlations.map((correlation, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-lg border ${getCorrelationBg(correlation.value)}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                        {correlation.icon === 'moon' && <Moon className="w-5 h-5 text-primary" />}
                        {correlation.icon === 'activity' && <Activity className="w-5 h-5 text-primary" />}
                        {correlation.icon === 'users' && <Heart className="w-5 h-5 text-primary" />}
                        {correlation.icon === 'sun' && <Sun className="w-5 h-5 text-primary" />}
                        {correlation.icon === 'clock' && <Clock className="w-5 h-5 text-primary" />}
                      </div>
                      <div>
                        <div className="font-medium">{correlation.factor1} vs. {correlation.factor2}</div>
                        <div className="text-sm text-muted-foreground">{correlation.description}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-bold ${getCorrelationColor(correlation.value)}`}>
                        {formatCorrelation(correlation.value)}
                      </div>
                      <div className="text-xs text-muted-foreground">correlation</div>
                    </div>
                  </div>

                  {/* Correlation Visualization */}
                  <div className="mt-3 h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        correlation.value >= 0 ? 'bg-success' : 'bg-destructive'
                      }`}
                      style={{
                        width: `${Math.abs(correlation.value) * 100}%`,
                        marginLeft: correlation.value < 0 ? `${50 + correlation.value * 50}%` : '50%',
                      }}
                    />
                  </div>

                  <div className="mt-3 p-3 bg-background/50 rounded">
                    <div className="flex items-start gap-2">
                      <Sparkles className="w-4 h-4 text-amber-500 mt-0.5" />
                      <span className="text-sm">{correlation.insight}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </TabsContent>

        {/* Patterns Tab */}
        <TabsContent value="patterns" className="space-y-4 mt-4">
          {progressReport && (
            <>
              {/* Weekly Rhythm */}
              <Card className="p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Calendar className="w-5 h-5 text-secondary" />
                  Weekly Rhythm Analysis
                </h3>
                
                <div className="space-y-4">
                  {progressReport.weeklyRhythm.map((day, idx) => (
                    <div key={idx} className="flex items-center gap-4">
                      <div className="w-16 text-sm font-medium">{day.day}</div>
                      <div className="flex-1 flex items-center gap-2">
                        <div className="flex-1">
                          <Progress value={day.avgMood * 10} className="h-2" />
                        </div>
                        <span className="text-sm text-muted-foreground w-12">
                          {day.avgMood.toFixed(1)} mood
                        </span>
                      </div>
                      {day.note && (
                        <Badge variant="outline" className="text-xs">
                          {day.note}
                        </Badge>
                      )}
                    </div>
                  ))}
                </div>
              </Card>

              {/* Seasonal Patterns */}
              <Card className="p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Sun className="w-5 h-5 text-amber-500" />
                  Seasonal Affect Patterns
                </h3>
                
                <div className="grid grid-cols-2 gap-4">
                  {progressReport.seasonalPatterns.map((season, idx) => (
                    <div key={idx} className="p-4 bg-muted/50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{season.season}</span>
                        <div className="flex items-center gap-1">
                          {season.trend === 'up' && <ArrowUpRight className="w-4 h-4 text-success" />}
                          {season.trend === 'down' && <ArrowDownRight className="w-4 h-4 text-destructive" />}
                        </div>
                      </div>
                      <div className="text-2xl font-bold mb-1">{season.avgMood.toFixed(1)}</div>
                      <div className="text-xs text-muted-foreground">{season.insight}</div>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Trigger Identification */}
              <Card className="p-6">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-orange-500" />
                  Trigger Identification
                </h3>
                
                <div className="space-y-3">
                  {progressReport.triggers.map((trigger, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full ${
                          trigger.impact === 'positive' ? 'bg-success' : 'bg-destructive'
                        }`} />
                        <span>{trigger.name}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={trigger.impact === 'positive' ? 'default' : 'destructive'}>
                          {trigger.impact === 'positive' ? '+' : ''}{trigger.effect}
                        </Badge>
                        <span className="text-xs text-muted-foreground">{trigger.metric}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Goals Tab */}
        <TabsContent value="goals" className="space-y-4 mt-4">
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <Target className="w-5 h-5 text-primary" />
                Custom Goal Tracking
              </h3>
              <Button size="sm">
                + Add Goal
              </Button>
            </div>

            <div className="space-y-4">
              {goals.map((goal) => (
                <div key={goal.id} className="p-4 rounded-lg border">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        goal.completed ? 'bg-success/20 text-success' : 'bg-primary/10 text-primary'
                      }`}>
                        {goal.icon === 'moon' && <Moon className="w-5 h-5" />}
                        {goal.icon === 'activity' && <Activity className="w-5 h-5" />}
                        {goal.icon === 'heart' && <Heart className="w-5 h-5" />}
                        {goal.icon === 'brain' && <Brain className="w-5 h-5" />}
                      </div>
                      <div>
                        <div className="font-medium">{goal.title}</div>
                        <div className="text-sm text-muted-foreground">{goal.description}</div>
                      </div>
                    </div>
                    {goal.completed && (
                      <Badge className="bg-success text-success-foreground">
                        ✓ Complete
                      </Badge>
                    )}
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Progress</span>
                      <span className="font-medium">
                        {goal.currentValue} / {goal.targetValue} {goal.unit}
                      </span>
                    </div>
                    <Progress 
                      value={(goal.currentValue / goal.targetValue) * 100} 
                      className="h-2"
                    />
                  </div>

                  <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                    <span>Started: {new Date(goal.startDate).toLocaleDateString()}</span>
                    <span>Target: {new Date(goal.targetDate).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Monthly Summary */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold flex items-center gap-2">
                <FileText className="w-5 h-5 text-secondary" />
                Monthly Wellness Summary
              </h3>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export Report
              </Button>
            </div>

            <div className="p-4 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-3 mb-4">
                <Calendar className="w-5 h-5 text-primary" />
                <span className="font-medium">January 2026 Summary</span>
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Total Check-ins</div>
                  <div className="font-semibold">28</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Consistency</div>
                  <div className="font-semibold">93%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Mood Improvement</div>
                  <div className="font-semibold text-success">+15%</div>
                </div>
                <div>
                  <div className="text-muted-foreground">Stress Reduction</div>
                  <div className="font-semibold text-success">-22%</div>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AnalyticsPortal;
