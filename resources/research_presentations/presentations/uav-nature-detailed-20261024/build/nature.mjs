export function author(p) {
 const s=p.slides.add(),K=2/3;
 const C={ink:'#26313B',gray:'#64717B',line:'#B9C2C8',teal:'#397C87',purple:'#87739D',rust:'#B26C52',green:'#61856D',blue:'#58799B',light:'#F7F9FA'};
 function rect(x,y,w,h,fill='#FFFFFF',stroke=C.line,r=4,dash=false){return s.shapes.add({geometry:r?'roundRect':'rect',position:{left:x*K,top:y*K,width:w*K,height:h*K},fill,line:{fill:stroke,width:1.3*K,style:dash?'dashed':'solid'},borderRadius:r*K});}
 function tx(v,x,y,w,h,size=20,col=C.ink,bold=false,align='left'){
 const a=s.shapes.add({geometry:'textbox',position:{left:x*K,top:y*K,width:w*K,height:h*K},fill:'none',line:{fill:'none',width:0}});a.text=v;a.text.style={fontSize:size*K,typeface:'PingFang SC',color:col,bold,alignment:align,verticalAlignment:'middle',insets:0,autoFit:'none',wrap:'square'};return a;}
 function edge(pts,col=C.ink,dash=false,arrow=true,width=1.8){for(let i=1;i<pts.length;i++){
 const [x,y]=pts[i-1],[xx,yy]=pts[i];s.shapes.add({geometry:'connector',kind:'straight',position:{left:Math.min(x,xx)*K,top:Math.min(y,yy)*K,width:Math.abs(xx-x)*K,height:Math.abs(yy-y)*K,horizontalFlip:xx<x,verticalFlip:yy<y},line:{fill:col,width:width*K,style:dash?'dashed':'solid'},...(i===pts.length-1&&arrow?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});}}
 function panel(letter,label,sub,x,y,w,h,color){rect(x,y,w,h,'#FFFFFF',C.line,6);edge([[x,y+58],[x+w,y+58]],color,false,false,2);tx(letter,x+14,y+9,30,34,29,C.ink,true);tx(label,x+51,y+8,w-65,35,26,color,true);tx(sub,x+51,y+40,w-65,17,13,C.gray);}
 function node(label,x,y,w,h,color=C.blue,fill='#F3F6F9',size=19){rect(x,y,w,h,fill,color,3);tx(label,x+7,y+3,w-14,h-6,size,C.ink,false,'center');}
 function dot(x,y,r,col,stroke=col){s.shapes.add({geometry:'ellipse',position:{left:(x-r)*K,top:(y-r)*K,width:2*r*K,height:2*r*K},fill:col,line:{fill:stroke,width:1*K}});}
 rect(0,0,1920,1080,'#FFFFFF','#FFFFFF',0);
 tx('无人机大小脑协同：可验证反馈与长期记忆',30,17,1575,52,39,C.ink,true);
 tx('高层语言规划与事后评价 · 条件向量化 · 冻结PPO策略执行',32,70,1450,27,20,C.gray);
 tx('拟新增：LLM Critic、反馈验证与RAG',1470,71,420,25,17,C.gray,false,'right');
 // Closed loop with verified feedback only.
 edge([[1665,145],[1665,112],[600,112],[600,145]],C.rust,true,true,2.2);
 tx('验证通过的反馈，用于下一轮计划修订',890,87,650,24,18,C.rust,false,'center');
 // a: environment, including physical entities and recorded transitions.
 panel('a','野火仿真环境','Wildfire environment',30,145,300,520,C.gray);
 rect(48,226,264,242,'#FCFDFD','#D4DADF',2);
 for(let x=70;x<310;x+=30)edge([[x,237],[x,457]],'#E4E9EB',false,false,.6);
 for(let y=247;y<464;y+=30)edge([[58,y],[302,y]],'#E4E9EB',false,false,.6);
 // Spatial objects are native diagram symbols, not an illustrative background.
 [[150,310,24],[185,318,30],[205,347,26],[173,355,28]].forEach(([x,y,r])=>dot(x,y,r,'#E4B89E','#C98E6C'));
 [[90,275],[260,290],[110,410],[255,405]].forEach(([x,y],i)=>{rect(x-12,y-12,24,24,'#EAF2F5',C.teal,2);tx('U'+(i+1),x-21,y-34,42,20,13,C.teal,true,'center');});
 [[140,385],[215,410],[232,355]].forEach(([x,y])=>dot(x,y,5,C.ink));
 rect(66,348,26,23,'#E1EDE3',C.green,0);rect(269,433,25,20,'#E1EDE3',C.green,0);
 edge([[64,243],[108,243]],C.gray,false,true,1.5);tx('风向',111,231,65,20,13,C.gray);
 tx('100×100栅格 / 4架无人机',50,478,260,24,17,C.gray);
 tx('火场    救援人员    充电站',50,507,260,24,17,C.gray);
 node('状态记录 sₜ',50,546,118,42,C.gray,'#F6F7F8',18);
 node('轨迹 τₜ:ₜ₊₂₀',185,546,125,42,C.gray,'#F6F7F8',18);
 tx('观测：火情、人员、链路、电量',50,603,260,24,17,C.ink);
 tx('结果：过火、认知率、AoI、耗尽',50,634,260,23,17,C.ink);
 // b: planner with input assembly, language reasoning and two structured output heads.
 panel('b','决策大模型','LLM task planner',380,145,440,370,C.teal);
 node('当前环境状态 sₜ  ＋  检索经验 Mₜ',399,221,402,43,C.teal,'#F1F6F6',20);
 edge([[600,264],[600,280]],C.teal);
 node('提示词组装与决策推理',444,280,312,50,C.teal,'#F1F6F6',22);
 tx('任务约束 / 资源预算 / 历史有效反馈',417,336,366,25,17,C.gray,false,'center');
 edge([[600,361],[600,373],[500,373],[500,386]],C.teal);
 edge([[600,373],[710,373],[710,386]],C.teal);
 rect(399,386,194,109,'#FFFFFF','#ADC3C7',3);rect(607,386,194,109,'#FFFFFF','#ADC3C7',3);
 tx('任务标签 ℓᵢ',413,394,166,26,20,C.teal,true);
 tx('探索 / 中继 / 充电\n按无人机逐一分配',413,427,166,59,17);
 tx('结构化建议 zᵢ',621,394,166,26,20,C.teal,true);
 tx('waypoints / 人员ID\n充电点 / 有效时窗',621,427,166,59,17);
 edge([[330,320],[380,320]],C.ink);tx('观测',332,285,47,24,15,C.gray,false,'center');
 // c: retrospective LLM evaluation uses observed outcomes and evidence, not just a scalar score.
 panel('c','执行后评价大模型','LLM Critic  /  拟新增',860,145,470,370,C.purple);
 node('决策前 sₜ\n原计划 pₜ',881,220,154,60,C.purple,'#F7F4F9',18);
 node('执行后 sₜ₊₂₀\n实际轨迹 τ',1155,220,154,60,C.purple,'#F7F4F9',18);
 edge([[1035,250],[1155,250]],C.purple);tx('对齐比较',1036,220,118,22,15,C.purple,false,'center');
 edge([[958,280],[958,294],[1095,294],[1095,305]],C.purple);
 edge([[1232,280],[1232,294],[1095,294]],C.purple,false,false);
 node('指标差分 Δm ＋ 实体 / 时间窗证据',881,305,428,42,C.purple,'#F7F4F9',20);
 tx('认知变化 / 通信覆盖 / AoI / 火情 / 电量',886,354,418,24,16,C.gray,false,'center');
 edge([[1095,379],[1095,392]],C.purple);
 node('LLM评价与建议生成',940,392,310,43,C.purple,'#F7F4F9',21);
 edge([[1095,435],[1095,449]],C.purple);
 node('反馈 fₜ = {评分 q, 证据 E, 修改 δp, 置信度 c}',881,449,428,45,C.purple,'#FFFFFF',17);
 // d: verifier is independent of the candidate evaluation, and can defer judgement.
 panel('d','反馈验证模型','独立判别器与指标核查  /  拟新增',1370,145,520,370,C.rust);
 node('证据核对：实体、时间窗与指标是否一致',1390,220,480,46,C.rust,'#FCF5F1',20);
 edge([[1630,266],[1630,279]],C.rust);
 node('执行归因：计划偏差 / 执行失败 / 安全接管',1390,279,480,46,C.rust,'#FCF5F1',19);
 edge([[1630,325],[1630,338]],C.rust);
 node('延迟收益判别，争议样本可触发配对分支回放',1390,338,480,46,C.rust,'#FCF5F1',19);
 edge([[1630,384],[1630,402]],C.rust);
 node('接受',1390,428,145,48,C.green,'#F2F7F2',21);
 node('待定',1558,428,145,48,C.gray,'#F6F7F8',21);
 node('驳回',1725,428,145,48,C.rust,'#FCF5F1',21);
 edge([[1630,402],[1462,402],[1462,428]],C.rust);edge([[1630,402],[1630,428]],C.rust);edge([[1630,402],[1797,402],[1797,428]],C.rust);
 tx('采用并入库',1390,481,145,21,16,C.green,false,'center');tx('延长观察',1558,481,145,21,16,C.gray,false,'center');tx('退回重评',1725,481,145,21,16,C.rust,false,'center');
 edge([[1330,360],[1370,360]],C.purple);tx('fₜ',1334,331,32,24,17,C.purple,false,'center');
 // Task contract adapter, frozen and deterministic.
 panel('e','任务条件向量化','Deterministic task adapter',380,555,440,140,C.teal);
 node('标签 ℓᵢ\n建议 zᵢ',399,624,97,49,C.teal,'#F1F6F6',16);
 node('目标更新\n短航迹转航点',527,624,127,49,C.teal,'#F1F6F6',16);
 node('拼接观测\noᵢ ∈ ℝ³⁵⁵',686,624,115,49,C.teal,'#F1F6F6',17);
 edge([[496,648],[527,648]],C.teal);edge([[654,648],[686,648]],C.teal);
 edge([[600,515],[600,555]],C.teal);tx('pₜ = {ℓᵢ, zᵢ}',612,522,180,26,18,C.teal);
 // f: current MAPPO implementation. Policy and centralized state-value network are distinct.
 panel('f','PPO低层控制：共享策略与中心化价值网络','当前代码采用MAPPO，部署时冻结策略网络',860,555,1030,460,C.blue);
 tx('策略网络 πθ（部署）',892,618,459,28,22,C.blue,true,'center');
 tx('价值网络 Vφ（仅训练）',1400,618,468,28,22,C.blue,true,'center');
 edge([[1375,621],[1375,943]],'#D7DEE4',true,false,1);
 const ax=904,aw=428,vx=1406,vw=452;
 node('局部观测与任务条件 oᵢ（355维）',ax,655,aw,32,C.blue,'#F1F5F9',18);
 node('自身 oᵢ ＋ 其他观测均值 ＋ 全局特征 g',vx,655,vw,32,C.blue,'#F1F5F9',17);
 node('LayerNorm(355)',ax,703,aw,30,C.blue,'#FFFFFF',18);
 node('LayerNorm(2 × obs_dim + global_dim)',vx,703,vw,30,C.blue,'#FFFFFF',17);
 node('Linear(355, 384)  +  ReLU  +  LayerNorm',ax,749,aw,36,C.blue,'#F1F5F9',18);
 node('Linear(input, 384)  +  ReLU  +  LayerNorm',vx,749,vw,36,C.blue,'#F1F5F9',17);
 node('Linear(384, 384)  +  ReLU  +  LayerNorm',ax,801,aw,36,C.blue,'#F1F5F9',18);
 node('Linear(384, 384)  +  ReLU  +  LayerNorm',vx,801,vw,36,C.blue,'#F1F5F9',18);
 node('Linear(384, 9)  /  action logits',ax,853,aw,31,C.blue,'#FFFFFF',18);
 node('Linear(384, 1)  /  状态价值 Vφ',vx,853,vw,31,C.blue,'#FFFFFF',18);
 node('Categorical  /  部署取 argmax  /  9类动作 aᵢ',ax,900,aw,36,C.blue,'#E8F0F7',18);
 node('回报与优势估计 GAE',vx,900,vw,36,C.blue,'#E8F0F7',19);
 for(let x of [1118,1632])for(let [y,yy] of [[687,703],[733,749],[785,801],[837,853],[884,900]])edge([[x,y],[x,yy]],C.blue);
 edge([[820,648],[879,648],[879,671],[904,671]],C.teal);
 tx('训练路径',883,963,92,24,16,C.gray,true);
 node('on-policy rollout',984,953,219,43,C.gray,'#F8F9FA',18);
 node('GAE与价值目标',1230,953,213,43,C.gray,'#F8F9FA',18);
 node('Lclip = E[min(rÂ, clip(r,1±ε)Â)]\n价值损失 Lᵥ，更新 θ 与 φ',1470,953,398,43,C.gray,'#F8F9FA',15);
 edge([[1203,975],[1230,975]],C.gray,true);edge([[1443,975],[1470,975]],C.gray,true);
 // g: concrete RAG pipeline, no externally curated expert library.
 panel('g','RAG长期经验记忆','仅收录通过验证的经验，保留适用条件与来源  /  拟新增',30,735,790,280,C.green);
 rect(49,814,217,152,'#FFFFFF','#AFC4B6',3);
 tx('经验记录 eₖ',62,822,191,27,21,C.green,true);
 tx('状态 s / 计划 p\n结果 Δm / 有效反馈 f\n证据 E / 置信度 c\n适用条件 κ / 来源 ID',62,857,191,98,16);
 rect(294,814,218,152,'#FFFFFF','#AFC4B6',3);
 tx('编码与向量索引',308,822,190,27,21,C.green,true);
 node('状态与经验编码器',308,861,190,37,C.green,'#F2F7F3',17);
 node('向量索引 ＋ 原始记录',308,916,190,34,C.green,'#F2F7F3',17);
 edge([[403,898],[403,916]],C.green);
 rect(540,814,260,152,'#FFFFFF','#AFC4B6',3);
 tx('条件过滤与 Top-k 检索',554,822,232,27,20,C.green,true);
 tx('查询 qₜ = 状态摘要 ＋ 目标缺口\n按资源、任务阶段过滤\n相似度与验证置信度重排',554,861,232,84,17);
 edge([[266,889],[294,889]],C.green);edge([[512,889],[540,889]],C.green);
 tx('输出：与当前场景相匹配的有效经验 Mₜ',49,978,750,24,19,C.green,true,'center');
 edge([[730,735],[730,712],[350,712],[350,450],[380,450]],C.green,true,true,2);
 tx('检索经验 Mₜ',368,706,260,25,16,C.green,false,'center');
 // Validation writes pass-only records to RAG, using an outer routing lane.
 edge([[1890,450],[1904,450],[1904,1030],[800,1030],[800,1015]],C.green,true,true,1.8);
 tx('仅接受的反馈写入记忆',1380,1014,420,22,15,C.green,false,'center');
 // The physical action loop and shared evidence bundle are separated from semantic feedback.
 edge([[904,918],[848,918],[848,1049],[16,1049],[16,610],[30,610]],C.ink,false,true,2);
 tx('动作执行与环境演化',390,1023,420,23,17,C.ink,false,'center');
 edge([[1230,555],[1230,535],[1095,535],[1095,515]],C.purple,true,true,1.8);
 edge([[1230,535],[1640,535],[1640,515]],C.purple,true,true,1.8);
 tx('20步执行证据包',881,519,190,17,14,C.purple,false,'center');
 // Rejected feedback is routed for revision before any memory write.
 edge([[1797,476],[1797,520],[1349,520],[1349,490],[1330,490]],C.rust,true,true,1.6);
 tx('部署执行',32,1055,130,20,14,C.ink);edge([[165,1065],[215,1065]],C.ink,false,true,1.8);
 tx('反馈 / 记忆',240,1055,150,20,14,C.gray);edge([[390,1065],[440,1065]],C.green,true,true,1.8);
 tx('设计图：评价、验证、RAG待实现与消融验证',1305,1055,585,20,14,C.gray,false,'right');
 s.speakerNotes.text='本图是现有系统与拟新增研究机制的详细设计，非已经实现的完整系统。现有PPO实现实际为MAPPO：lowlevel/models.py定义共享Actor，输入355维，输入LayerNorm后两层384维Linear+ReLU+LayerNorm，最后Linear到9个离散动作logits，部署使用argmax。CentralizedCritic将自身观测、其他无人机观测均值与全局特征拼接，采用同样两层MLP并输出单个状态价值。experiments/lowlevel_training/train.py使用on-policy rollout、GAE和PPO clipped策略目标以及clipped Huber价值损失，价值目标归一化。此处高层LLM Critic与训练期价值网络是不同模块。任务编码按现有task_spec与route_converter接口，短航迹逐点转换为当前航点，目标完成更新，不能虚构额外训练的embedding网络。Critic拟接收同一20步执行窗口的前后状态、原计划、实际轨迹，输出分数、证据、修改建议与置信度。验证器拟采用独立判别器加指标核查，核查证据与时间窗、执行偏差、安全接管及延迟收益；有可重放快照时，可对争议样本进行同状态配对分支回放。相关性并不等于因果归因，验证器应允许待定/拒答，不能保证所有反馈正确。仅通过验证的反馈入可复用RAG记忆，待定延长观察，驳回重评；RAG存系统自身经过验证的经验，不含专家经验库。RAG拟按场景条件过滤后进行相似度和验证置信度重排。示意场景实体位置仅作图示，不是实验观测。设计基于用户要求及项目本地代码，未引入未经验证的性能数据。';
 return s;
}
