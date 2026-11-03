export function author(presentation) {
 if (presentation.slides.items.length) throw new Error('Refusing to change an existing deck');
 const s=presentation.slides.add();
 const K=0.8;
 const C={ink:'#172A40',muted:'#586A7E',teal:'#087E8B',purple:'#7654AD',orange:'#C86A27',blue:'#276AB3',gray:'#718296',green:'#328568'};
 function box(x,y,w,h,fill,stroke,r=18){return s.shapes.add({geometry:'roundRect',position:{left:x*K,top:y*K,width:w*K,height:h*K},fill,line:{fill:stroke,width:2*K},borderRadius:r*K});}
 function text(str,x,y,w,h,size=22,color=C.ink,bold=false,align='left'){
 const sh=s.shapes.add({geometry:'textbox',position:{left:x*K,top:y*K,width:w*K,height:h*K},fill:'none',line:{fill:'none',width:0}});sh.text=str;sh.text.style={fontSize:size*K,typeface:'PingFang SC',color,bold,alignment:align,verticalAlignment:'middle',insets:0,autoFit:'none',wrap:'square'};return sh;}
 function line(points,color,width=2.5,dashed=false,arrow=true){
  for(let i=1;i<points.length;i++){
   const [x,y]=points[i-1],[xx,yy]=points[i];
   const sh=s.shapes.add({geometry:'connector',kind:'straight',position:{left:Math.min(x,xx)*K,top:Math.min(y,yy)*K,width:Math.abs(xx-x)*K,height:Math.abs(yy-y)*K,horizontalFlip:xx<x,verticalFlip:yy<y},line:{fill:color,width:width*K,style:dashed?'dashed':'solid'},...(i===points.length-1&&arrow?{tail:{type:'triangle',width:'med',length:'med'}}:{})});
  }
 }
 function title(a,b,x,y,w,color){text(a,x+20,y+16,w-40,34,27,color,true);text(b,x+20,y+54,w-40,28,17,C.muted);}
 box(0,0,1600,900,'#FFFFFF','#FFFFFF',0);
 text('面向无人机协同的可验证反馈与长期记忆框架',40,30,1470,58,42,C.ink,true);
 text('决策大模型 × 评价大模型 × 反馈验证 × RAG × PPO低层控制',42,91,1450,30,23,C.muted);
 // Verified feedback returns to the next decision, rather than gating an unexecuted plan.
 line([[1380,220],[1380,165],[580,165],[580,220]],C.orange,3,true);
 text('通过验证的反馈 → 下一轮计划修订',715,128,510,30,21,C.orange,true,'center');
 // Environment.
 box(40,220,310,380,'#F3F6F9','#8A99A8');
 title('野火仿真环境','状态观测 / 动作执行 / 任务指标',40,220,310,C.ink);
 box(62,317,266,160,'#FFFFFF','#CCD5DF',10);
 for(let x=90;x<320;x+=38)line([[x,327],[x,467]],'#E5EBF1',1,false,false);
 for(let y=345;y<470;y+=30)line([[72,y],[318,y]],'#E5EBF1',1,false,false);
 box(150,349,78,48,'#FBB673','#EC8E36',10);text('火区',150,356,78,30,19,'#8C3D0E',true,'center');
 [[100,355],[255,415],[122,435],[280,345]].forEach(([x,y],i)=>{box(x-13,y-13,26,26,'#DEF0F6',C.teal,6);text(String(i+1),x-10,y-11,20,22,16,C.teal,true,'center');});
 box(76,393,42,24,'#DBEBD9','#75A26F',3);text('充电',76,394,42,20,12,'#466340',false,'center');
 text('100×100栅格 / 无人机 / 救援人员',63,490,264,30,16,C.muted);
 text('火情 / 链路 / 电量 / 人员认知',63,526,264,30,18,C.muted);
 text('真实环境状态与执行轨迹',63,562,264,25,20,C.ink,true);
 // Planner.
 box(420,220,320,260,'#EAF7F7',C.teal);
 title('决策大模型','高层任务规划与结构化建议',420,220,320,C.teal);
 text('任务标签',442,312,276,29,22,C.teal,true);
 text('探索感知 / 通信中继 / 充电',442,347,276,28,20);
 text('可操作建议',442,389,276,29,22,C.teal,true);
 text('航点或短航迹 / 人员ID / 充电点',442,424,276,35,19);
 line([[350,350],[420,350]],C.ink,3);
 text('观测',351,311,66,28,18,C.muted,false,'center');
 // LLM critic.
 box(810,220,320,260,'#F3EFF9',C.purple);
 title('评价大模型 · Critic','对执行结果进行事后评价',810,220,320,C.purple);
 text('输入：前后状态、原计划\n与实际执行轨迹',832,311,276,52,19);
 text('输出：分数 ＋ 证据 ＋ 改进方向',832,367,276,43,19,C.purple,true);
 text('“哪里出了问题，下一步怎么改”',832,428,276,28,18,C.muted);
 // Validation model.
 box(1200,220,360,260,'#FFF3E9',C.orange);
 title('反馈验证模型','验证证据、执行归因与时间尺度',1200,220,360,C.orange);
 text('① 指令是否被实际执行？',1222,315,316,29,21);
 text('② 收益是否尚未显现？',1222,354,316,29,21);
 text('③ 证据是否支持批评与建议？',1222,393,316,29,21);
 text('通过→采用   待定→追踪   驳回→重评',1222,438,316,26,16,C.orange,true);
 line([[1130,350],[1200,350]],C.purple,3);
 text('候选反馈',1129,306,74,31,16,C.purple,false,'center');
 // Raw candidate corrections, kept away from validated memory.
 line([[1260,480],[1260,505],[1040,505],[1040,480]],C.purple,2,true);
 text('驳回：退回重评',1070,484,170,23,16,C.purple,false,'center');
 // Task encoder and PPO execution pathway.
 line([[700,480],[700,560],[945,560],[945,650]],C.teal,3);
 text('任务标签 ＋ 结构化建议',715,525,300,28,20,C.teal,true);
 box(810,650,270,150,'#EDF6FB',C.blue);
 title('任务条件向量化','确定性编码，与现有接口兼容',810,650,270,C.blue);
 text('环境观测 ＋ 任务条件',830,747,230,27,20,C.ink,true,'center');
 box(1160,650,400,150,'#EDF4FC',C.blue);
 text('PPO低层控制',1180,665,360,32,27,C.blue,true);
 text('部署：冻结策略 πθ → 9类离散动作',1180,710,360,29,22,C.ink,true);
 text('训练期：价值网络 Vφ / rollout / PPO更新',1180,758,360,25,17,C.muted);
 line([[1080,725],[1160,725]],C.blue,3);
 text('条件向量',1080,687,80,28,16,C.blue,false,'center');
 // Post-execution trace delivered to the evaluator.
 line([[1480,650],[1480,582],[1100,582],[1100,480]],C.purple,2.5,true);
 text('执行20步：前后状态与实际轨迹',1140,589,352,30,20,C.purple,true,'center');
 line([[1480,582],[1480,480]],C.purple,2.5,true);
 text('同步执行证据',1335,541,140,28,18,C.purple,false,'right');
 // Memory retrieval, with only validated lessons admitted to the reusable memory.
 box(420,650,320,150,'#EFF8F2',C.green);
 title('RAG长期记忆','检索与当前状态相似的历史经验',420,650,320,C.green);
 text('已验证的执行经验 / 有效反馈',442,747,276,28,20,C.green,true);
 line([[580,650],[580,480]],C.green,3,true);
 text('检索经验',590,587,130,29,19,C.green,true);
 line([[1560,360],[1582,360],[1582,833],[580,833],[580,800]],C.green,2.5,true);
 text('验证通过后写入：状态—计划—结果—证据—改进建议',755,811,685,28,19,C.green,true,'center');
 // Physical action loop.
 line([[1360,800],[1360,875],[195,875],[195,600]],C.ink,3);
 text('低层动作 → 环境演化 → 下一轮观测',260,846,665,29,21,C.ink,true,'center');
 text('研究设计：评价、验证与RAG为拟新增模块；验证不通过的反馈不进入可复用记忆。',42,180,1120,28,18,C.muted);
 s.speakerNotes.text='此页为研究架构设计。现有PPO训练得到的低层策略部署时冻结，任务标签与建议采用确定性向量化。PPO训练期保留价值网络与on-policy rollout，但不使用双Q、Target Q或off-policy经验回放。高层Critic使用另一个大模型在执行20步后结合原计划、前后状态和实际轨迹给出分数、证据及可操作修改建议。反馈验证模型需要结合结构化指标核查与独立判别，区分规划错误、执行偏差和延迟收益；单纯覆盖率下降不能证明计划错误。通过验证的反馈可修订下一轮计划并形成RAG经验，待定反馈继续观察，驳回反馈退回重评。RAG为系统自身执行与反馈形成的经验，不设置专家经验库。评价、反馈验证、RAG均是拟新增研究模块，效果需要后续消融验证。';
 return {slideCount:1,shapes:s.shapes.items.length};
}
