
% breakyaxes splits data in an axes so that data is in a low and high pane.
%
%   breakYAxes(splitYLim) splitYLim is a 2 element vector containing a range
%   of y values from splitYLim(1) to splitYLim(2) to remove from the axes.
%   They must be within the current yLimis of the axes.
%
%   breakYAxes(splitYLim,splitHeight) splitHeight is the distance to 
%   seperate the low and high side.  Units are the same as 
%   get(AX,'uints') default is 0.015
% 
%   breakYAxes(splitYLim,splitHeight,xOverhang) xOverhang stretches the 
%   axis split graphic to extend past the top and bottom of the plot by
%   the distance set by XOverhang.  Units are the same as get(AX,'units')
%   default value is 0.015
%
%   breakYAxes(AX, ...) performs the operation on the axis specified by AX
%
function breakInfo = breakyaxis(varargin)

    %Validate Arguements
    if nargin < 1 || nargin > 4
       error('Wrong number of arguements'); 
    end

    if isscalar(varargin{1}) && ishandle(varargin{1})
        mainAxes = varargin{1};
        argOffset = 1;
        argCnt = nargin - 1;
        if ~strcmp(get(mainAxes,'Type'),'axes')
           error('Handle object must be Type Axes'); 
        end
    else
        mainAxes = gca;
        argOffset = 0;
        argCnt = nargin;
    end
    
    if (strcmp(get(mainAxes,'XScale'),'log'))
        error('Log X Axes are not supported'); 
    end
    
    if (argCnt < 3)
        xOverhang = 0.015;
    else
        xOverhang = varargin{3 + argOffset};
        if  numel(xOverhang) ~= 1 || ~isreal(xOverhang) || ~isnumeric(xOverhang)
            error('XOverhang must be a scalar number');
        elseif (xOverhang < 0)
            error('XOverhang must not be negative');
        end
        xOverhang = double(xOverhang);
    end
    
    if (argCnt < 2)
        splitHeight = 0.015;
    else
        splitHeight = varargin{2 + argOffset};
        if  numel(xOverhang) ~= 1 || ~isreal(xOverhang) || ~isnumeric(xOverhang)
            error('splitHeight must be a scalar number');
        elseif (xOverhang < 0)
            error('splitHeight must not be negative');
        end
        splitHeight = double(splitHeight);
    end
    
    splitYLim = varargin{1 + argOffset};
    if numel(splitYLim) ~= 2 || ~isnumeric(splitYLim) || ~isreal(xOverhang)
       error(splitYLim,'Must be a vector length 2');
    end
    splitYLim = double(splitYLim);
    
    mainYLim = get(mainAxes,'YLim');
    if (any(splitYLim >= mainYLim(2)) || any(splitYLim <= mainYLim(1)))
       error('splitYLim must be in the range given by get(AX,''YLim'')');
    end
    
    mainPosition = get(mainAxes,'Position');
    if (splitHeight > mainPosition(3) ) 
       error('Split width is too large') 
    end
   
    %We need to create 4 axes
    % lowAxes - is used for the low y axis and low pane data
    % highAxes - is used to the high y axis and high pane data
    % annotationAxes - is used to display the x axis and title
    % breakAxes - this is an axes with the same size and position as main
    %   is it used to draw a seperator between the low and high side
    

    %Grab Some Parameters from the main axis (e.g the one we are spliting)
    mainYLim = get(mainAxes,'YLim');
    mainXLim = get(mainAxes,'XLim');
    mainPosition = get(mainAxes,'Position');
    mainParent = get(mainAxes,'Parent');
    mainHeight = mainPosition(4); %Positions have the format [low bottom width height]
    %mainYRange = mainYLim(2) - mainYLim(1);
    mainFigure = get(mainAxes,'Parent');
    mainXColor = get(mainAxes,'XColor');
    mainLineWidth = get(mainAxes,'LineWidth');
    figureColor = get(mainFigure,'Color');
    mainXTickLabelMode = get(mainAxes,'XTickLabelMode');
    mainYLabel = get(mainAxes,'YLabel');
    mainYDir = get(mainAxes,'YDir');
    mainLayer = get(mainAxes,'Layer');
    
    %Save Main Axis Z Order
    figureChildren = get(mainFigure,'Children');
    zOrder = find(figureChildren == mainAxes);
    
    %Calculate where axesLow and axesHigh will be layed on screen
    %And their respctive YLimits
    lowYLimTemp = [mainYLim(1) splitYLim(1)];
    highYLimTemp = [splitYLim(2) mainYLim(2)];

    lowYRangeTemp = lowYLimTemp(2) - lowYLimTemp(1);
    highYRangeTemp = highYLimTemp(2) - highYLimTemp(1);

    lowHeightTemp = lowYRangeTemp / (lowYRangeTemp + highYRangeTemp) * (mainHeight - splitHeight);
    highHeightTemp = highYRangeTemp / (lowYRangeTemp + highYRangeTemp) * (mainHeight - splitHeight);

    lowStretch = (lowHeightTemp + splitHeight/2) / lowHeightTemp;
    lowYRange = lowYRangeTemp * lowStretch;
    lowHeight = lowHeightTemp * lowStretch;

    highStretch = (highHeightTemp + splitHeight/2) / highHeightTemp;
    highYRange = highYRangeTemp * highStretch;
    highHeight = highHeightTemp * highStretch;
    
    lowYLim = [mainYLim(1) mainYLim(1)+lowYRange];
    highYLim = [mainYLim(2)-highYRange mainYLim(2)];
    
    if (strcmp(mainYDir, 'normal')) 
        lowPosition = mainPosition;
        lowPosition(4) = lowHeight; 

        highPosition = mainPosition;    %(!!!) look here for position indices!
        highPosition(2) = mainPosition(2) + lowHeight;
        highPosition(4) = highHeight;
    else
        %Low Axis will actually go on the high side a vise versa
        highPosition = mainPosition;
        highPosition(4) = highHeight; 

        lowPosition = mainPosition;
        lowPosition(2) = mainPosition(2) + highHeight;
        lowPosition(4) = lowHeight;
    end
 
    %Create the Annotations layer, if the Layer is top, draw the axes on
    %top (e.g. after) drawing the low and high pane
    if strcmp(mainLayer,'bottom')
        annotationAxes = CreateAnnotaionAxes(mainAxes,mainParent)
    end
    
    %Create and position the lowAxes. Remove all X Axis Annotations, the 
    %title, and a potentially offensive tick mark 
    lowAxes = copyobj(mainAxes,mainParent);
    set(lowAxes,'Position', lowPosition, ...
        'YLim', lowYLim, ... 
        'XLim', mainXLim, ...
        'XGrid' ,'off', ...
        'XMinorGrid', 'off', ...
        'XMinorTick','off', ...
        'XTick', [], ...
        'XTickLabel', [], ...
        'box','off');
    if strcmp(mainLayer,'bottom')
        set(lowAxes,'Color','none');
    end
    delete(get(lowAxes,'XLabel')); 
    delete(get(lowAxes,'YLabel'));
    delete(get(lowAxes,'Title'));
    
    if strcmp(mainXTickLabelMode,'auto')
        yTick =  get(lowAxes,'YTick');
        set(lowAxes,'YTick',yTick(1:(end-1)));
    end
    
    %Create and position the highAxes. Remove all X Axis annotations, the 
    %title, and a potentially offensive tick mark 
    highAxes = copyobj(mainAxes,mainParent);
    set(highAxes,'Position', highPosition, ...
        'YLim', highYLim, ...
        'XLim', mainXLim, ...
        'XGrid' ,'off', ...
        'XMinorGrid', 'off', ...
        'XMinorTick','off', ...
        'XTick', [], ...
        'XTickLabel', [], ...
        'box','off');
    if strcmp(mainLayer,'bottom') %(!!!) is it only about layers?
        set(highAxes,'Color','none');
    end
    delete(get(highAxes,'XLabel')); 
    delete(get(highAxes,'YLabel'));
    delete(get(highAxes,'Title'));
    
    if strcmp(mainXTickLabelMode,'auto')
        yTick =  get(highAxes,'YTick');
        set(highAxes,'YTick',yTick(2:end));
    end

        %Create the Annotations layer, if the Layer is top, draw the axes on
    %top (e.g. after) drawing the low and high pane
    if strcmp(mainLayer,'top')
        annotationAxes = CreateAnnotaionAxes(mainAxes,mainParent);
        set(annotationAxes, 'Color','none');
    end
    
    %Create breakAxes, remove all graphics objects and hide all annotations
    breakAxes = copyobj(mainAxes,mainParent);
    children = get(breakAxes,'Children');
    for i = 1:numel(children)
       delete(children(i)); 
    end
    
    set(breakAxes,'Color','none');
    %Stretch the breakAxes horizontally to cover the vertical axes lines
    orignalUnits = get(breakAxes,'Units');
    set(breakAxes,'Units','Pixel');
    breakPosition = get(breakAxes,'Position');
    nudgeFactor = get(breakAxes,'LineWidth');
    breakPosition(3) = breakPosition(3) +  nudgeFactor;
    set(breakAxes,'Position',breakPosition);
    set(breakAxes,'Units',orignalUnits);

    %Stretch the breakAxes horizontally to create an overhang for sylistic
    %effect
    breakPosition = get(breakAxes,'Position');
    breakPosition(1) = breakPosition(1) - xOverhang;
    breakPosition(3) = breakPosition(3) +  2*xOverhang;
    set(breakAxes,'Position',breakPosition);
    
    %Create a sine shaped patch to seperate the 2 sides
    breakYLim = [mainPosition(2) mainPosition(2)+mainPosition(4)];
    set(breakAxes,'ylim',breakYLim);
    theta = linspace(0,2*pi,100);
    xPoints = linspace(mainXLim(1),mainXLim(2),100);
    amp = splitHeight/2 * 0.9;
    yPoints1 = amp * sin(theta) + mainPosition(2) + lowHeightTemp;
    yPoints2 = amp * sin(theta) + mainPosition(2) + mainPosition(4) - highHeightTemp;
    patchPointsY = [yPoints1 yPoints2(end:-1:1) yPoints1(1)];
    patchPointsX = [xPoints  xPoints(end:-1:1)  xPoints(1)];
    patch(patchPointsX,patchPointsY ,figureColor,'EdgeColor',figureColor,'Parent',breakAxes); %use of pathc(!!!)?

    %Create A Line To Delineate the low and high edge of the patch
    line('yData',yPoints1,'xdata',xPoints,'Parent',breakAxes,'Color',mainXColor,'LineWidth',mainLineWidth);
    line('yData',yPoints2,'xdata',xPoints,'Parent',breakAxes,'Color',mainXColor,'LineWidth',mainLineWidth);

    set(breakAxes,'Visible','off');
    
    %Make the old main axes invisiable
    invisibleObjects = RecursiveSetVisibleOff(mainAxes);

    %Preserve the z-order of the figure
    uistack([lowAxes highAxes breakAxes annotationAxes],'down',zOrder-1)
    
    %Set the rezise mode to position so that we can dynamically change the
    %size of the figure without screwing things up
    set([lowAxes highAxes breakAxes annotationAxes],'ActivePositionProperty','Position');
 
    %Playing with the titles labels etc can cause matlab to reposition
    %the axes in some cases.  Mannually force the position to be correct. 
    set([breakAxes annotationAxes],'Position',mainPosition);
    
    %Save the axes so we can unbreak the axis easily
    breakInfo = struct();
    breakInfo.lowAxes = lowAxes;
    breakInfo.highAxes = highAxes;
    breakInfo.breakAxes = breakAxes;
    breakInfo.annotationAxes = annotationAxes;
    breakInfo.invisibleObjects = invisibleObjects;
end

function list = RecursiveSetVisibleOff(handle) 
    list = [];
    list = SetVisibleOff(handle,list);
    
end 

function list = SetVisibleOff(handle, list)
    if (strcmp(get(handle,'Visible'),'on'))
        set(handle,'Visible','off');
        list = [list handle];
    end
    
    children = get(handle,'Children');
    for i = 1:numel(children)
        list = SetVisibleOff(children(i),list);
    end
end
    
function annotationAxes = CreateAnnotaionAxes(mainAxes,mainParent)

    %Create Annotation Axis, Remove graphics objects, YAxis annotations
    %(except YLabel) and make background transparent
    annotationAxes = copyobj(mainAxes,mainParent);
    
    set(annotationAxes,'XLimMode','Manual');
    
    children = get(annotationAxes,'Children');
    for i = 1:numel(children)
       delete(children(i)); 
    end

    %Save the yLabelpostion because it will move when we delete yAxis
    %ticks
    yLabel = get(annotationAxes,'YLabel');
    yLabelPosition = get(yLabel,'Position');
    
    set(annotationAxes,'YGrid' ,'off', ...
        'YMinorGrid', 'off', ...
        'YMinorTick','off', ...
        'YTick', [], ...
        'YTickLabel', []);
    
    %Restore the pevious label postition
    set(yLabel,'Position',yLabelPosition);
end


