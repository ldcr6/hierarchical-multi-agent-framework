<template>
  <el-container class="page">
    <el-header class="header">
      <div class="title">
        <span class="logo">HMAS</span>
        基于多智能体协同的个性化辅助编程与自动化测试系统
        <span class="sub">演示版 · Planner / Coder / Verifier 三层框架 v0</span>
      </div>
      <div class="env" v-if="env">
        <el-tag :type="env.docker ? 'success' : (env.local_runner ? 'warning' : 'danger')" effect="dark" size="small">
          {{ env.docker ? 'Docker 沙箱已就绪' : (env.local_runner ? '无 Docker（仅离线演示）' : 'Docker 未启动') }}
        </el-tag>
        <el-tag :type="keyReady ? 'success' : 'info'" effect="dark" size="small">
          {{ keyReady ? 'DeepSeek 密钥已配置' : '未配置模型密钥' }}
        </el-tag>
      </div>
    </el-header>

    <el-main>
      <el-row :gutter="16">
        <!-- ============ left: input ============ -->
        <el-col :span="8">
          <el-card shadow="never" class="block">
            <template #header><b>1. 提交题目</b></template>
            <el-form label-position="top" size="default">
              <el-form-item label="题目来源">
                <el-radio-group v-model="form.source">
                  <el-radio-button value="example">内置示例</el-radio-button>
                  <el-radio-button value="humaneval">函数签名 + 文档字符串</el-radio-button>
                  <el-radio-button value="mbpp">自然语言描述</el-radio-button>
                </el-radio-group>
              </el-form-item>

              <template v-if="form.source === 'example'">
                <el-form-item label="选择示例">
                  <el-select v-model="form.exampleId" style="width: 100%" @change="onExample">
                    <el-option v-for="e in examples" :key="e.id" :label="e.title" :value="e.id" />
                  </el-select>
                </el-form-item>
                <div v-if="currentExample" class="hint">{{ currentExample.note }}</div>
                <pre v-if="currentExample" class="code small">{{ currentExample.prompt.trim() }}</pre>
              </template>

              <template v-else-if="form.source === 'humaneval'">
                <el-form-item label="函数签名和文档字符串（文档里的 >>> 示例会自动作为题面测试）">
                  <el-input v-model="form.prompt" type="textarea" :rows="10" class="mono"
                            placeholder="def add(a: int, b: int) -> int:&#10;    &quot;&quot;&quot;返回 a 与 b 的和。&#10;    >>> add(1, 2)&#10;    3&#10;    &quot;&quot;&quot;" />
                </el-form-item>
              </template>

              <template v-else>
                <el-form-item label="需求描述（可以附上 assert 示例，每行一条）">
                  <el-input v-model="form.prompt" type="textarea" :rows="7"
                            placeholder="编写函数 ...&#10;assert f(1) == 2" />
                </el-form-item>
                <el-form-item label="入口函数名">
                  <el-input v-model="form.entryPoint" placeholder="例如 top_k_words" class="mono" />
                </el-form-item>
              </template>

              <el-form-item>
                <template #label>
                  验收测试（可选）
                  <el-tooltip content="框架运行时看不到这些测试，只在最后用来检验最终代码，相当于“隐藏测试”。" placement="top">
                    <el-icon class="q"><QuestionFilled /></el-icon>
                  </el-tooltip>
                </template>
                <el-input v-model="form.acceptance" type="textarea" :rows="3" class="mono"
                          placeholder="assert f(10) == 20" />
              </el-form-item>

              <el-form-item label="运行模式">
                <el-radio-group v-model="form.mode">
                  <el-radio value="offline">离线演示（脚本回放，不调用模型）</el-radio>
                  <el-radio value="live">真实模型（付费 API）</el-radio>
                </el-radio-group>
              </el-form-item>
              <template v-if="form.mode === 'live'">
                <el-form-item label="模型">
                  <el-select v-model="form.model" style="width: 100%">
                    <el-option v-for="m in (env?.models || [])" :key="m.name" :value="m.name"
                               :label="m.name + (m.has_key ? '' : '（未配置密钥）')" :disabled="!m.has_key" />
                  </el-select>
                </el-form-item>
                <el-checkbox v-model="form.confirmPaid">我确认本次运行会调用付费 API（每题约十几次调用）</el-checkbox>
              </template>
              <el-alert v-if="form.mode === 'offline' && form.source !== 'example'" type="warning" :closable="false"
                        show-icon title="离线演示只能运行内置示例；自定义题目请选择“真实模型”。" class="mt" />

              <el-button type="primary" class="run-btn" :loading="submitting" @click="submit">开始运行</el-button>
            </el-form>
          </el-card>

          <el-card shadow="never" class="block">
            <template #header><b>运行记录</b></template>
            <el-empty v-if="!history.length" description="还没有运行记录" :image-size="60" />
            <div v-for="h in history" :key="h.id" class="hist" :class="{ active: run && run.id === h.id }"
                 @click="open(h.id)">
              <div class="hist-title">{{ h.title }}</div>
              <div class="hist-meta">
                <el-tag size="small" :type="statusType(h.status, h.passed)">{{ statusText(h.status, h.passed) }}</el-tag>
                <span>{{ h.model }}</span><span>{{ fmtTime(h.created) }}</span>
              </div>
            </div>
          </el-card>
        </el-col>

        <!-- ============ right: run ============ -->
        <el-col :span="16">
          <el-card v-if="!run" shadow="never" class="block welcome">
            <h3>工作流程</h3>
            <ol>
              <li><b>Planner（规划者）</b> 把题目分解成 1–5 个辅助函数，并给每个子任务打难度分。</li>
              <li><b>Coder（编码者）</b> 按依赖顺序逐个实现辅助函数，最后用已验证的辅助函数组合出入口函数。</li>
              <li><b>Verifier（验证者）</b> 只看需求、不看代码，为每个函数编写单元测试，连同题面示例一起放进 Docker 沙箱执行。</li>
              <li>测试失败时，把失败用例和实际输出反馈给 Coder 修复（最多 2 次），保留得分最高的版本。</li>
            </ol>
            <p class="hint">在左侧选择示例或输入题目，点击“开始运行”。</p>
          </el-card>

          <template v-else>
            <el-card shadow="never" class="block">
              <div class="run-head">
                <div>
                  <div class="run-title">{{ run.title }}</div>
                  <div class="hist-meta">
                    <el-tag :type="statusType(run.status, rootPassed)" effect="dark">{{ statusText(run.status, rootPassed) }}</el-tag>
                    <span>模型：{{ run.model }}</span>
                    <span>沙箱：{{ run.sandbox === 'docker' ? 'Docker（断网、只读、资源限制）' : '本地进程（仅开发用）' }}</span>
                    <span>用时 {{ run.elapsed }} 秒</span>
                  </div>
                </div>
              </div>
              <el-alert v-if="run.error" type="error" :title="run.error" :closable="false" show-icon class="mt" />
              <el-steps :active="phase" finish-status="success" align-center class="mt">
                <el-step title="任务分解" description="Planner" />
                <el-step title="子任务编码与验证" description="Coder + Verifier" />
                <el-step title="组合入口函数" description="Coder + Verifier" />
                <el-step title="完成" />
              </el-steps>
            </el-card>

            <el-row :gutter="16">
              <el-col :span="12">
                <el-card shadow="never" class="block">
                  <template #header><b>2. 任务分解树</b></template>
                  <el-empty v-if="!run.nodes.length" description="等待 Planner 分解…" :image-size="60" />
                  <div v-else class="tree">
                    <div class="tnode root" :class="[rootNode.status, { sel: selected === rootNode.name }]"
                         @click="selected = rootNode.name">
                      <div class="tname">{{ rootNode.name }} <span class="badge">入口函数</span></div>
                      <div class="tmeta">
                        难度 <el-rate :model-value="stars(rootNode.complexity)" disabled size="small" />
                        <el-tag size="small" :type="nodeTagType(rootNode.status)">{{ nodeStatusText(rootNode) }}</el-tag>
                      </div>
                    </div>
                    <div class="children">
                      <div v-for="n in helpers" :key="n.name" class="tnode" :class="[n.status, { sel: selected === n.name }]"
                           @click="selected = n.name">
                        <div class="tname mono">{{ n.signature.replace(/^def /, '') }}</div>
                        <div class="tdesc">{{ n.description }}</div>
                        <div class="tmeta">
                          难度 <el-rate :model-value="stars(n.complexity)" disabled size="small" />
                          <el-tag size="small" :type="nodeTagType(n.status)">{{ nodeStatusText(n) }}</el-tag>
                        </div>
                        <div v-if="n.depends_on.length" class="tdesc">依赖：{{ n.depends_on.join('、') }}</div>
                      </div>
                      <div v-if="!helpers.length" class="hint">Planner 没有给出可用的子任务，直接整体实现。</div>
                    </div>
                    <div v-if="run.plan?.composition" class="compose"><b>组合方式：</b>{{ run.plan.composition }}</div>
                    <div v-if="run.plan?.problems?.length" class="hint">计划检查：{{ run.plan.problems.join('；') }}</div>
                  </div>
                </el-card>
              </el-col>

              <el-col :span="12">
                <el-card shadow="never" class="block">
                  <template #header><b>3. 执行过程</b><span class="hint"> （共 {{ llmSteps }} 次模型调用）</span></template>
                  <div class="timeline-box">
                    <el-timeline>
                      <el-timeline-item v-for="(t, i) in run.trace" :key="i" :type="traceType(t)" :hollow="!t.role">
                        <div class="tl-text" @click="t.node && (selected = t.node)">
                          <span class="role" :class="t.role || 'system'">{{ t.role ? roleName[t.role] : '沙箱' }}</span>{{ traceText(t) }}
                        </div>
                        <div v-if="t.failures?.length" class="fails">
                          <div v-for="(f, j) in t.failures" :key="j" class="mono">✗ {{ f }}</div>
                        </div>
                        <details v-if="t.response" class="resp">
                          <summary>查看模型输出（输入 {{ t.prompt_tokens }} / 输出 {{ t.completion_tokens }} tokens）</summary>
                          <pre class="code small">{{ t.response }}</pre>
                        </details>
                      </el-timeline-item>
                      <el-timeline-item v-if="running" type="primary" :icon="Loading" class="spin">运行中…</el-timeline-item>
                    </el-timeline>
                  </div>
                </el-card>
              </el-col>
            </el-row>

            <el-card v-if="selNode" shadow="never" class="block">
              <template #header>
                <b>4. 验证与测试结果：<span class="mono">{{ selNode.name }}</span></b>
                <span class="hint"> 验证等级 L{{ selNode.level }}（静态检查 + 沙箱测试）·
                  题面示例 {{ selNode.visible_tests.length }} 条 · 生成测试 {{ selNode.generated_tests.length }} 条
                  <template v-if="selNode.confidence != null"> · Coder 自评置信度 {{ selNode.confidence }}</template>
                </span>
              </template>
              <el-empty v-if="!selNode.candidates.length" description="还没有代码版本" :image-size="60" />
              <el-tabs v-else v-model="candTab">
                <el-tab-pane v-for="(c, i) in selNode.candidates" :key="i" :name="String(i)">
                  <template #label>
                    <span :class="c.passed ? 'ok' : 'bad'">{{ c.passed ? '✓' : '✗' }}</span>
                    {{ originText(c.origin) }}
                    <el-tag v-if="selNode.final_index === i" size="small" type="success">最终采用</el-tag>
                  </template>
                  <el-row :gutter="12">
                    <el-col :span="11"><pre class="code">{{ c.code }}</pre></el-col>
                    <el-col :span="13">
                      <el-alert v-if="c.static_error" type="error" :closable="false" show-icon
                                :title="'静态检查未通过：' + c.static_error" />
                      <el-alert v-else-if="c.load_error || c.sandbox_error" type="error" :closable="false" show-icon
                                :title="c.load_error || c.sandbox_error" />
                      <template v-else-if="c.tests_total != null">
                        <div class="score">
                          通过 <b>{{ c.tests_passed }}</b> / {{ c.tests_total }}
                          <el-progress :percentage="pct(c.tests_passed, c.tests_total)" :status="c.passed ? 'success' : 'exception'"
                                       :stroke-width="10" style="flex: 1" />
                        </div>
                        <el-table :data="c.results" size="small" border max-height="320">
                          <el-table-column label="测试" min-width="200">
                            <template #default="{ row }"><span class="mono">{{ row.test }}</span></template>
                          </el-table-column>
                          <el-table-column label="来源" width="88">
                            <template #default="{ row }">
                              <el-tag size="small" :type="row.kind === 'visible' ? 'warning' : 'info'">
                                {{ row.kind === 'visible' ? '题面示例' : '生成测试' }}</el-tag>
                            </template>
                          </el-table-column>
                          <el-table-column label="结果" width="150">
                            <template #default="{ row }">
                              <span v-if="row.ok" class="ok">✓ 通过</span>
                              <span v-else class="bad mono">✗ {{ row.error }}</span>
                            </template>
                          </el-table-column>
                        </el-table>
                      </template>
                    </el-col>
                  </el-row>
                </el-tab-pane>
              </el-tabs>
            </el-card>

            <el-row :gutter="16">
              <el-col :span="14">
                <el-card shadow="never" class="block">
                  <template #header>
                    <b>5. 最终代码</b>
                    <el-button v-if="run.final_code" size="small" text type="primary" @click="copy(run.final_code)">复制</el-button>
                  </template>
                  <el-empty v-if="!run.final_code" :description="running ? '运行中…' : '没有生成代码'" :image-size="60" />
                  <pre v-else class="code">{{ run.final_code }}</pre>
                  <template v-if="run.acceptance_results">
                    <div class="sub-title">验收测试（框架运行时不可见）：
                      <b :class="run.acceptance_results.passed === run.acceptance_results.total && !run.acceptance_results.load_error ? 'ok' : 'bad'">
                        {{ run.acceptance_results.passed }} / {{ run.acceptance_results.total }} 通过</b>
                    </div>
                    <el-alert v-if="run.acceptance_results.load_error || run.acceptance_results.sandbox_error" type="error"
                              :closable="false" :title="run.acceptance_results.load_error || run.acceptance_results.sandbox_error" />
                    <el-table :data="run.acceptance_results.results" size="small" border>
                      <el-table-column label="测试"><template #default="{ row }"><span class="mono">{{ row.test }}</span></template></el-table-column>
                      <el-table-column label="结果" width="200">
                        <template #default="{ row }">
                          <span v-if="row.ok" class="ok">✓ 通过</span><span v-else class="bad mono">✗ {{ row.error }}</span>
                        </template>
                      </el-table-column>
                    </el-table>
                  </template>
                </el-card>
              </el-col>
              <el-col :span="10">
                <el-card shadow="never" class="block">
                  <template #header><b>6. 资源消耗</b></template>
                  <el-table v-if="run.usage" :data="usageRows" size="small" border>
                    <el-table-column prop="role" label="角色" width="90" />
                    <el-table-column prop="calls" label="调用" width="56" />
                    <el-table-column prop="prompt_tokens" label="输入 tokens" />
                    <el-table-column prop="completion_tokens" label="输出 tokens" />
                  </el-table>
                  <div v-if="run.usage && run.usage.total.calls" class="share">
                    <div v-for="r in usageRows.slice(0, 3)" :key="r.role" class="share-row">
                      <span>{{ r.role }}</span>
                      <el-progress :percentage="tokenShare(r)" :stroke-width="8" style="flex: 1" />
                    </div>
                    <div class="hint">token 占比；缓存命中 {{ run.usage.total.cached_calls }} 次（重复运行同一题不再计费）</div>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </template>
        </el-col>
      </el-row>
    </el-main>
  </el-container>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading, QuestionFilled } from '@element-plus/icons-vue'

const env = ref(null)
const examples = ref([])
const history = ref([])
const run = ref(null)
const selected = ref(null)
const candTab = ref('0')
const submitting = ref(false)
let timer = null

const form = reactive({
  source: 'example', exampleId: null, prompt: '', entryPoint: '', acceptance: '',
  mode: 'offline', model: 'deepseek-flash', confirmPaid: false,
})

async function api(path, opts = {}) {
  const r = await fetch('/api' + path, { headers: { 'Content-Type': 'application/json' }, ...opts })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || `请求失败（${r.status}）`)
  return data
}

const currentExample = computed(() => examples.value.find(e => e.id === form.exampleId))
const keyReady = computed(() => (env.value?.models || []).some(m => m.has_key && m.name.startsWith('deepseek')))
const running = computed(() => run.value && ['queued', 'running'].includes(run.value.status))
const rootNode = computed(() => run.value?.nodes.find(n => n.root))
const helpers = computed(() => run.value?.nodes.filter(n => !n.root) || [])
const rootPassed = computed(() => {
  const s = rootNode.value?.status
  return s === 'passed' ? true : s === 'failed' ? false : null
})
const selNode = computed(() => run.value?.nodes.find(n => n.name === selected.value))
const llmSteps = computed(() => (run.value?.trace || []).filter(t => t.role).length)
const phase = computed(() => {
  const r = run.value
  if (!r || !r.plan) return 0
  if (r.status === 'done') return 4
  const root = rootNode.value
  return root && root.status === 'running' ? 2 : 1
})
const roleName = { planner: 'Planner', coder: 'Coder', verifier: 'Verifier' }
const usageRows = computed(() => {
  const u = run.value?.usage
  if (!u) return []
  const rows = Object.entries(u.by_role).map(([k, v]) => ({ role: roleName[k] || k, ...v }))
  rows.push({ role: '合计', ...u.total })
  return rows
})

function onExample() {
  const e = currentExample.value
  if (e) form.acceptance = e.acceptance
}

function stars(k) { return k == null ? 0 : Math.round(k * 4 + 1) }
function pct(a, b) { return b ? Math.round((100 * a) / b) : 0 }
function tokenShare(r) {
  const t = run.value.usage.total
  const all = t.prompt_tokens + t.completion_tokens
  return all ? Math.round((100 * (r.prompt_tokens + r.completion_tokens)) / all) : 0
}
function fmtTime(ts) { return ts ? new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false }) : '' }

function statusText(s, passed) {
  if (s === 'queued') return '排队中'
  if (s === 'running') return '运行中'
  if (s === 'error') return '出错'
  return passed === true ? '完成 · 验证通过' : passed === false ? '完成 · 验证未全部通过' : '完成'
}
function statusType(s, passed) {
  if (s === 'running' || s === 'queued') return 'primary'
  if (s === 'error') return 'danger'
  return passed === false ? 'warning' : 'success'
}
function nodeTagType(s) { return { passed: 'success', failed: 'danger', running: 'primary', pending: 'info' }[s] }
function nodeStatusText(n) {
  const tries = n.candidates.length
  if (n.status === 'passed') return tries > 1 ? `通过（修复 ${tries - 1} 次）` : '一次通过'
  if (n.status === 'failed') return '未全部通过'
  if (n.status === 'running') return '进行中'
  return '等待'
}
function originText(o) {
  if (o === 'implement') return '首稿'
  if (o === 'compose') return '组合'
  const m = /^repair-(\d+)$/.exec(o)
  return m ? `修复 ${m[1]}` : o
}
function traceType(t) {
  if (t.step === 'verify') return t.passed ? 'success' : 'danger'
  if (t.step === 'plan_check') return 'warning'
  return 'primary'
}
function traceText(t) {
  const n = t.node ? `「${t.node}」` : ''
  switch (t.step) {
    case 'plan': return `将题目分解为 ${run.value.plan?.n_helpers ?? '?'} 个辅助函数，并评估难度`
    case 'plan_check': return `计划检查：${(t.problems || []).join('；')}`
    case 'implement': return `实现${n}`
    case 'compose': return `用已验证的辅助函数组合入口函数${n}`
    case 'repair': return `根据失败用例修复${n}`
    case 'write_tests': return `只看需求、不看代码，为${n}编写单元测试`
    case 'verify': {
      const what = `验证${n}的${originText(t.origin)}`
      if (t.static_error) return `${what}：静态检查未通过（${t.static_error}）`
      if (t.load_error || t.sandbox_error) return `${what}：运行出错（${t.load_error || t.sandbox_error}）`
      if (t.tests_total == null) return `${what}：${t.passed ? '通过' : '未通过'}`
      return `${what}：沙箱测试 ${t.tests_passed}/${t.tests_total} 通过`
    }
    default: return t.step
  }
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); ElMessage.success('已复制') } catch { ElMessage.error('复制失败') }
}

async function loadHistory() { history.value = await api('/runs').catch(() => []) }

async function poll(id) {
  clearTimeout(timer)
  try {
    const d = await api('/runs/' + id)
    if (!run.value || run.value.id !== id) return
    const prevNode = selected.value
    run.value = d
    // follow the node being worked on while running; keep the user's choice afterwards
    if (['queued', 'running'].includes(d.status)) {
      const cur = d.nodes.find(n => n.status === 'running')
      if (cur) selected.value = cur.name
      timer = setTimeout(() => poll(id), 1000)
    } else {
      if (!prevNode || !d.nodes.some(n => n.name === prevNode)) selected.value = rootNode.value?.name
      loadHistory()
    }
  } catch (e) {
    ElMessage.error(e.message)
  }
}

async function open(id) {
  run.value = { id, nodes: [], trace: [], status: 'queued', title: '加载中…' }
  selected.value = null
  await poll(id)
  if (!['queued', 'running'].includes(run.value.status)) selected.value = rootNode.value?.name
}

watch(selNode, (n, old) => {
  if (!n) return
  if (!old || old.name !== n.name || Number(candTab.value) >= n.candidates.length) {
    candTab.value = String(n.final_index ?? Math.max(n.candidates.length - 1, 0))
  } else if (running.value) {
    candTab.value = String(n.candidates.length - 1)
  }
})

async function submit() {
  const body = { mode: form.mode, acceptance: form.acceptance, model: form.model, confirm_paid: form.confirmPaid }
  if (form.source === 'example') {
    if (!form.exampleId) return ElMessage.warning('请选择一个示例')
    body.example_id = form.exampleId
  } else {
    body.style = form.source
    body.prompt = form.prompt
    body.entry_point = form.entryPoint
  }
  submitting.value = true
  try {
    const { id } = await api('/runs', { method: 'POST', body: JSON.stringify(body) })
    await loadHistory()
    await open(id)
  } catch (e) {
    ElMessage.error(e.message)
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  env.value = await api('/status').catch(() => null)
  examples.value = await api('/examples').catch(() => [])
  if (examples.value.length) { form.exampleId = examples.value[0].id; onExample() }
  loadHistory()
})
</script>
