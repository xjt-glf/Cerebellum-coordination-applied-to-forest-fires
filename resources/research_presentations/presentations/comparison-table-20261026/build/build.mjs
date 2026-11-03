import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const base=path.resolve(process.argv[2]);
const data=JSON.parse(await fs.readFile(path.join(base,'build','data.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const t=s.tables.add({rows:data.values.length,columns:4,left:56,top:66,width:1168,height:588,columnWidths:[454,238,238,238],values:data.values});
t.styleOptions={headerRow:false,bandedRows:false};
t.borders.assign({fill:'#D9DFE3',width:.65,style:'solid'});
for(let r=0;r<data.values.length;r++){
 t.rows[r].height=r===0?66:58;
 for(let c=0;c<4;c++){
  const cell=t.getCell(r,c);
  cell.fill=r===0?'#EDF2F4':r%2?'#FFFFFF':'#F7F9FA';
  cell.text.style={fontSize:r===0?(c===3?22:23):(c===0?23:22),typeface:'PingFang SC',bold:r===0,color:r===0?'#263E50':'#34424D',alignment:c===0?'left':'center',verticalAlignment:'middle',insets:{left:c===0?18:8,right:8,top:7,bottom:7},autoFit:'none',wrap:'none'};
  if(r>0&&data.winners[r-1].includes(c)){cell.text.style={fontSize:22,typeface:'PingFang SC',bold:true,color:'#287E77',alignment:'center',verticalAlignment:'middle',insets:8,autoFit:'none',wrap:'none'};}
 }
}
s.speakerNotes.text='来源：research_outputs/comprehensive_comparison_3seeds_20261022/metrics_summary.csv、metrics_by_episode.csv、report.md。三组均使用种子730000—730002，每局600步，相同冻结Actor与每20步规划协议。显示各局指标的均值±样本标准差，比例指标的标准差也以百分数展示。加权人员火情认知率按真实燃烧格·步加权，各局先计算再跨局平均。AoI P95先在每局人员时间样本取95分位，再跨局取均值和标准差；单位为仿真步。青绿色加粗表示相应指标均值最优，并列均标示，不能作为统计显著性标记。Codex为带工具、持续上下文与既有Actor知识的agent方案，千问为API方案，有效信息组织与工具预算不相等。n=3为描述性配对结果，不能孤立归因为模型品牌或参数能力。选用原单航点Codex组，与完整三组报告一致，没有混入后续短航迹实验。页面按用户要求仅展示对比数据表，不添加标题、结论、来源说明等额外正文。';
const img=await p.export({slide:s,format:'png',scale:2});
await fs.writeFile(path.join(base,'output','三种规划方法_数据对比.png'),new Uint8Array(await img.arrayBuffer()));
const candidatePath=path.join(base,'build','candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidatePath);
const skill='/Users/xiangjutao/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools','artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:base,candidatePath,finalPath:path.join(base,'output','启发式_千问_Codex_数据对比.pptx'),pythonExecutable:'/Users/xiangjutao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools','inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools','inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--require-native-table-slide','1'],explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[1],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:['PingFang SC']},verifyArtifactToolImport:true,receiptPath:path.join(base,'build','validation.json')});
console.log(JSON.stringify({file:result.finalPath,package:result.packageIntegrity.status,layout:result.presentationLayout.finding_count,import:result.firstPartyImport.passed}));
