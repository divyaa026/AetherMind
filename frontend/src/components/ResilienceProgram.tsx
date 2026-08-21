import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import MockDataBanner from '@/components/MockDataBanner';
import {
  Brain,
  BookOpen,
  CheckCircle,
  Lock,
  Play,
  Clock,
  Trophy,
  Target,
  Zap,
  Heart,
  Sparkles,
  ChevronRight,
  ArrowRight,
  Star,
  GraduationCap,
  RotateCcw
} from 'lucide-react';
import { toast } from '@/hooks/use-toast';
import AetherMindAPIService, { ResilienceWeek, ResilienceLesson, ResilienceExercise } from '@/services/AetherMindAPI';

interface ResilienceProgramProps {
  onNavigate?: (tab: string) => void;
}

const ResilienceProgram: React.FC<ResilienceProgramProps> = ({ onNavigate }) => {
  const [currentWeek, setCurrentWeek] = useState(1);
  const [programData, setProgramData] = useState<ResilienceWeek[]>([]);
  const [currentLesson, setCurrentLesson] = useState<ResilienceLesson | null>(null);
  const [resilienceScore, setResilienceScore] = useState(0);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<'overview' | 'lesson' | 'exercise'>('overview');

  const weekDescriptions = {
    1: "Emotional Awareness Foundation",
    2: "Emotional Awareness Foundation",
    3: "Stress Response Retraining",
    4: "Stress Response Retraining",
    5: "Cognitive Flexibility",
    6: "Cognitive Flexibility",
    7: "Resilience Integration",
    8: "Resilience Integration"
  };

  useEffect(() => {
    loadProgramData();
  }, []);

  const loadProgramData = async () => {
    setLoading(true);
    try {
      const [weeks, score] = await Promise.all([
        AetherMindAPIService.getResilienceProgram(),
        AetherMindAPIService.getResilienceScore()
      ]);
      setProgramData(weeks);
      setResilienceScore(score);
    } catch (error) {
      console.error('Failed to load program data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStartLesson = (lesson: ResilienceLesson) => {
    setCurrentLesson(lesson);
    setView('lesson');
  };

  const handleCompleteLesson = async (lessonId: string) => {
    try {
      await AetherMindAPIService.completeResilienceLesson(lessonId);
      toast({
        title: "Lesson completed! 🎉",
        description: "Great progress on your resilience journey.",
      });
      loadProgramData();
      setView('overview');
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to save progress",
        variant: "destructive"
      });
    }
  };

  const getWeekProgress = (week: ResilienceWeek) => {
    const totalLessons = week.lessons.length;
    const completedLessons = week.lessons.filter(l => l.completed).length;
    return Math.round((completedLessons / totalLessons) * 100);
  };

  const getOverallProgress = () => {
    if (programData.length === 0) return 0;
    const totalLessons = programData.reduce((acc, week) => acc + week.lessons.length, 0);
    const completedLessons = programData.reduce(
      (acc, week) => acc + week.lessons.filter(l => l.completed).length, 
      0
    );
    return Math.round((completedLessons / totalLessons) * 100);
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-8 bg-muted rounded w-1/3"></div>
          <div className="h-32 bg-muted rounded"></div>
        </div>
      </Card>
    );
  }

  // Lesson View
  if (view === 'lesson' && currentLesson) {
    return (
      <div className="space-y-6">
        <Card className="p-6">
          <Button variant="ghost" onClick={() => setView('overview')} className="mb-4">
            ← Back to Program
          </Button>
          
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
              {currentLesson.type === 'video' && <Play className="w-6 h-6 text-primary" />}
              {currentLesson.type === 'reading' && <BookOpen className="w-6 h-6 text-primary" />}
              {currentLesson.type === 'exercise' && <Target className="w-6 h-6 text-primary" />}
            </div>
            <div>
              <h2 className="text-xl font-semibold">{currentLesson.title}</h2>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Clock className="w-4 h-4" />
                {currentLesson.duration}
                <Badge variant="outline" className="ml-2">
                  {currentLesson.type}
                </Badge>
              </div>
            </div>
          </div>

          {/* Lesson Content */}
          <div className="prose prose-sm max-w-none">
            {currentLesson.type === 'video' && (
              <div className="aspect-video bg-muted rounded-lg flex items-center justify-center mb-6">
                <div className="text-center">
                  <Play className="w-16 h-16 text-primary mx-auto mb-2" />
                  <p className="text-muted-foreground">Video: {currentLesson.title}</p>
                </div>
              </div>
            )}
            
            <div className="space-y-4">
              <h3 className="font-semibold">Key Concepts</h3>
              <ul className="space-y-2">
                {currentLesson.keyPoints?.map((point, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <CheckCircle className="w-4 h-4 text-success mt-1 flex-shrink-0" />
                    <span>{point}</span>
                  </li>
                ))}
              </ul>

              {currentLesson.content && (
                <div className="p-4 bg-muted/50 rounded-lg mt-4">
                  <p className="text-sm">{currentLesson.content}</p>
                </div>
              )}
            </div>
          </div>

          {/* Practice Exercise */}
          {currentLesson.exercise && (
            <Card className="mt-6 p-4 border-secondary/20 bg-secondary/5">
              <h4 className="font-semibold flex items-center gap-2 mb-3">
                <Zap className="w-4 h-4 text-secondary" />
                Practice Exercise
              </h4>
              <p className="text-sm text-muted-foreground mb-4">
                {currentLesson.exercise.instructions}
              </p>
              <Button 
                variant="secondary"
                onClick={() => onNavigate?.(currentLesson.exercise?.navigateTo || 'breathing')}
              >
                Start Exercise
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </Card>
          )}

          <div className="mt-8 flex justify-end">
            <Button 
              size="lg"
              onClick={() => handleCompleteLesson(currentLesson.id)}
            >
              <CheckCircle className="w-5 h-5 mr-2" />
              Mark as Complete
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  // Overview View
  return (
    <div className="space-y-6">
      <MockDataBanner feature="Resilience Program" storageType="static" />
      
      {/* Header */}
      <Card className="p-6 bg-gradient-to-br from-primary/10 via-secondary/5 to-background">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <GraduationCap className="w-6 h-6 text-primary" />
            <div>
              <h2 className="text-xl font-semibold">8-Week Resilience Program</h2>
              <p className="text-sm text-muted-foreground">
                Build lasting mental strength through evidence-based practices
              </p>
            </div>
          </div>
        </div>

        {/* Overall Progress */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Overall Progress</span>
            <span className="font-semibold">{getOverallProgress()}%</span>
          </div>
          <Progress value={getOverallProgress()} className="h-3" />
        </div>

        {/* Resilience Score */}
        <div className="mt-6 p-4 bg-background/50 rounded-lg border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                <Star className="w-6 h-6 text-white" />
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Your Resilience Score</div>
                <div className="text-2xl font-bold">{resilienceScore}</div>
              </div>
            </div>
            <Badge variant="outline" className="text-xs">
              +12 this week
            </Badge>
          </div>
        </div>
      </Card>

      {/* Week Tabs */}
      <Tabs value={currentWeek.toString()} onValueChange={(v) => setCurrentWeek(parseInt(v))}>
        <ScrollArea className="w-full">
          <TabsList className="w-full justify-start gap-1 h-auto p-1 flex-nowrap">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((week) => {
              const weekData = programData.find(w => w.weekNumber === week);
              const isLocked = weekData?.locked;
              const progress = weekData ? getWeekProgress(weekData) : 0;
              
              return (
                <TabsTrigger 
                  key={week}
                  value={week.toString()}
                  disabled={isLocked}
                  className="relative flex-shrink-0 px-4 py-2"
                >
                  <div className="flex flex-col items-center gap-1">
                    <span className="text-xs">Week {week}</span>
                    {isLocked ? (
                      <Lock className="w-3 h-3 text-muted-foreground" />
                    ) : progress === 100 ? (
                      <Trophy className="w-3 h-3 text-amber-500" />
                    ) : (
                      <span className="text-xs text-muted-foreground">{progress}%</span>
                    )}
                  </div>
                </TabsTrigger>
              );
            })}
          </TabsList>
        </ScrollArea>

        {/* Week Content */}
        {programData.map((week) => (
          <TabsContent key={week.weekNumber} value={week.weekNumber.toString()} className="mt-4">
            <Card className="p-6">
              <div className="flex items-center gap-3 mb-6">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                  getWeekProgress(week) === 100 
                    ? 'bg-success/20 text-success' 
                    : 'bg-primary/10 text-primary'
                }`}>
                  {getWeekProgress(week) === 100 
                    ? <Trophy className="w-6 h-6" />
                    : <Brain className="w-6 h-6" />
                  }
                </div>
                <div>
                  <h3 className="text-lg font-semibold">
                    Week {week.weekNumber}: {weekDescriptions[week.weekNumber as keyof typeof weekDescriptions]}
                  </h3>
                  <p className="text-sm text-muted-foreground">{week.description}</p>
                </div>
              </div>

              {/* Lessons */}
              <div className="space-y-3">
                {week.lessons.map((lesson, idx) => (
                  <div
                    key={lesson.id}
                    className={`flex items-center gap-4 p-4 rounded-lg border transition-all ${
                      lesson.completed 
                        ? 'bg-success/5 border-success/20' 
                        : lesson.locked 
                          ? 'bg-muted/50 border-muted' 
                          : 'hover:border-primary/30 cursor-pointer'
                    }`}
                    onClick={() => !lesson.locked && !lesson.completed && handleStartLesson(lesson)}
                  >
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                      lesson.completed 
                        ? 'bg-success/20 text-success' 
                        : lesson.locked
                          ? 'bg-muted text-muted-foreground'
                          : 'bg-primary/10 text-primary'
                    }`}>
                      {lesson.completed ? (
                        <CheckCircle className="w-5 h-5" />
                      ) : lesson.locked ? (
                        <Lock className="w-5 h-5" />
                      ) : (
                        <span className="font-semibold">{idx + 1}</span>
                      )}
                    </div>
                    
                    <div className="flex-1">
                      <div className="font-medium">{lesson.title}</div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="w-3 h-3" />
                        {lesson.duration}
                        <Badge variant="outline" className="text-xs ml-1">
                          {lesson.type}
                        </Badge>
                      </div>
                    </div>
                    
                    {!lesson.locked && !lesson.completed && (
                      <ChevronRight className="w-5 h-5 text-muted-foreground" />
                    )}
                  </div>
                ))}
              </div>

              {/* Week Completion */}
              {getWeekProgress(week) === 100 && (
                <div className="mt-6 p-4 bg-success/10 rounded-lg border border-success/20">
                  <div className="flex items-center gap-3">
                    <Trophy className="w-6 h-6 text-success" />
                    <div>
                      <div className="font-semibold text-success">Week {week.weekNumber} Complete!</div>
                      <p className="text-sm text-muted-foreground">
                        Great job! You've mastered the fundamentals of {weekDescriptions[week.weekNumber as keyof typeof weekDescriptions]}.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      {/* Quick Actions */}
      <Card className="p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-amber-500" />
          Quick Practice
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <Button 
            variant="outline" 
            className="h-auto py-4 flex flex-col items-center gap-2"
            onClick={() => onNavigate?.('breathing')}
          >
            <Heart className="w-6 h-6 text-primary" />
            <span>Breathing Exercise</span>
          </Button>
          <Button 
            variant="outline"
            className="h-auto py-4 flex flex-col items-center gap-2"
            onClick={() => onNavigate?.('journal')}
          >
            <BookOpen className="w-6 h-6 text-secondary" />
            <span>Reflection Journal</span>
          </Button>
        </div>
      </Card>
    </div>
  );
};

export default ResilienceProgram;
