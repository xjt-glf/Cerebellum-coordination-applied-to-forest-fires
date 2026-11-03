export function author(p){
 const s=p.slides.add();s.background.fill='#FFFFFF';
 const C={ink:'#183B61',blue:'#2869A4',mid:'#5686B1',light:'#EEF5FB',line:'#B5CDDF',gray:'#597086'};
 function box(x,y,w,h,fill='#FFFFFF',stroke=C.line,r=3){return s.shapes.add({geometry:r?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:1.1},borderRadius:r});}
 function tx(v,x,y,w,h,size=20,color=C.ink,bold=false,align='left'){const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=v;a.text.style={fontSize:size,typeface:'PingFang SC',color,bold,alignment:align,verticalAlignment:'middle',insets:0,autoFit:'none',wrap:'square'};return a;}
 function edge(pts,dash=false,arrow=true,color=C.blue,width=1.6){for(let i=1;i<pts.length;i++){const[x,y]=pts[i-1],[xx,yy]=pts[i];s.shapes.add({geometry:'connector',kind:'straight',position:{left:Math.min(x,xx),top:Math.min(y,yy),width:Math.abs(xx-x),height:Math.abs(yy-y),horizontalFlip:xx<x,verticalFlip:yy<y},line:{fill:color,width,style:dash?'dashed':'solid'},...(arrow&&i===pts.length-1?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});}}
 function node(v,x,y,w,h,size=18,fill=C.light){box(x,y,w,h,fill);tx(v,x+8,y+4,w-16,h-8,size,C.ink,false,'center');}
 tx('执行感知与跨时间尺度反事实 Critic',40,25,1185,53,36,C.ink,true);
 tx('面向冻结PPO执行器的高层决策归因与可操作修正',42,83,1100,29,21,C.gray);
 edge([[40,120],[1240,120]],false,false,C.blue,2);
 // Pain point: an observed change admits different explanations.
 tx('01  痛点：短窗变化难以归因',42,141,330,35,24,C.ink,true);
 node('计划 pₜ → 执行20步 → 覆盖率下降',42,193,310,52,18);
 tx('相同现象，可能对应三种原因',43,260,310,29,19,C.gray);
 const causes=[['规划失误','撤走关键中继，造成通信缺口','修改任务分配'],['执行偏差','阻挡或安全接管，指令未落实','定位执行问题'],['收益延迟','正在转场，短期下降但后续获益','延长观察窗口']];
 causes.forEach(([a,b,c],i)=>{const y=304+i*78;edge([[43,y+65],[351,y+65]],false,false,'#D9E5EE',.8);tx(a,44,y,100,28,21,C.blue,true);tx(b,44,y+31,308,25,17,C.gray);tx(c,156,y+2,194,25,18,C.ink,false,'right');});
 tx('风险：错误反馈进入RAG后被反复复用',43,548,310,25,16,C.blue,true);
 edge([[372,145],[372,570]],false,false,'#D1DFEA',1);
 // Detailed architecture.
 tx('02  方法：意图—执行对齐与反事实收益预测',397,141,838,35,24,C.ink,true);
 tx('结构化输入',400,188,158,25,18,C.gray,true,'center');
 tx('关系与时序编码',585,188,171,25,18,C.gray,true,'center');
 tx('跨模态对齐',783,188,164,25,18,C.gray,true,'center');
 tx('可训练的预测分支',977,188,238,25,18,C.gray,true,'center');
 box(397,230,161,89,C.light);tx('环境关系图 Gₜ',406,237,143,29,20,C.ink,true,'center');tx('无人机、人员、火区\n通信与可达关系',405,269,145,41,15,C.gray,false,'center');
 node('图编码器\nGraph Transformer',586,230,170,89,16);
 box(397,355,161,89,C.light);tx('计划与执行轨迹',405,363,145,28,19,C.ink,true,'center');tx('原计划 p / 修改 p′\n执行轨迹 τₜ:ₜ₊₂₀',405,394,145,41,16,C.gray,false,'center');
 node('时序编码器\nTemporal Transformer',586,355,170,89,16);
 edge([[558,275],[586,275]]);edge([[558,400],[586,400]]);
 node('意图—执行\n交叉注意力\nCross-attn.',785,285,159,111,19,'#E2EDF7');
 edge([[756,275],[770,275],[770,310],[785,310]]);edge([[756,400],[770,400],[770,371],[785,371]]);
 box(976,229,238,66,C.light);tx('执行一致性分支',985,234,220,27,20,C.blue,true);tx('目标进度 / 执行偏差 / 接管',985,265,220,23,16,C.gray);
 box(976,312,238,66,C.light);tx('多时间尺度收益分支',985,317,220,27,20,C.blue,true);tx('h = 20、60、120步及整局',985,348,220,23,16,C.gray);
 box(976,395,238,66,C.light);tx('局部修改价值与不确定性',985,400,220,27,18,C.blue,true);tx('预测 Δmₕ(p→p′)，允许拒答',985,431,220,23,16,C.gray);
 edge([[944,315],[959,315],[959,262],[976,262]]);
 edge([[944,340],[959,340],[959,345],[976,345]]);
 edge([[944,367],[959,367],[959,428],[976,428]]);
 edge([[1214,262],[1230,262],[1230,477],[850,477],[850,495]]);
 edge([[1214,345],[1230,345]],false,false);edge([[1214,428],[1230,428]],false,false);
 box(397,495,817,72,'#20578A','#20578A',3);
 tx('评分大模型：依据预测与执行证据生成结构化修正',411,502,789,27,22,'#FFFFFF',true,'center');
 tx('修改对象 / 任务参数 / 改进方向 / 生效条件 / 置信度',411,535,789,24,18,'#FFFFFF',false,'center');
 // Offline supervision and validation, separated from deployment inference.
 edge([[40,588],[1240,588]],false,false,C.line,1);
 tx('03  离线监督',42,608,165,31,22,C.ink,true);
 node('同一快照分叉\n原计划 p / 局部修改 p′',213,606,270,54,17);
 node('冻结同一PPO与后续协议\n比较各时间尺度真实指标',520,606,300,54,17);
 node('训练执行与收益预测分支\nL = L执行 + λL收益 + μL排序',857,606,357,54,17);
 edge([[483,633],[520,633]],true);edge([[820,633],[857,633]],true);
 tx('验证重点：归因准确率、修改收益、误判率及去除各分支的消融；方案待实验验证',42,678,1185,24,16,C.gray);
 s.speakerNotes.text='本页是拟开展研究的方法设计，不表示已实现或已证实创新性。痛点源于当前项目中短窗任务指标与整局收益不一致、实际执行会受阻挡和充电保护影响。构建环境关系图以及计划/实际执行前缀的时序表示，通过交叉注意力进行意图—执行对齐。三个学习分支分别估计执行一致性、多时间尺度结果及局部任务修改的干预收益和不确定性。关系图的注意力权重不能直接称为因果解释。候选修改可以是目标、角色和交接时间；同状态分叉时保持同一冻结Actor和明确定义的后续控制协议，记录团队完整演化。训练标签保留过火、认知率、AoI等多维结果。20步时仅允许读取20步以内的轨迹，60/120步和整局结果只用于离线监督与留出集评估，不作为在线可见输入。损失中的λ、μ是待验证的超参数；本图不指定任意数值。LLM Critic负责依据证据生成对象、参数、方向、生效条件和置信度等结构化反馈。后续验证机制决定是否采用反馈并写入RAG，此页聚焦Critic内部，不展开整个系统。反事实收益限定于给定仿真、执行器和后续控制协议，不能直接外推为真实世界普遍因果效应。反事实方法已有相关工作，创新需比较COMA https://arxiv.org/abs/1705.08926 、C3 https://arxiv.org/abs/2603.06859 、CAR https://arxiv.org/abs/2606.08275 ，差异化候选是冻结低层执行器下的执行感知、跨时间尺度与可操作任务修正。文献链接来自本对话前序检索，本页没有新增文献结论或实验结果。';
 return s;
}
