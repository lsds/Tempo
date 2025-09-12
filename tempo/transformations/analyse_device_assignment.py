from tempo.core import tensor_ops as top
from tempo.core.datatypes import OpId
from tempo.core.dependence_graph import PDG
from tempo.core.device import DeviceGroup, device
from tempo.core.dl_backends import DLBackendName
from tempo.transformations.compilation_pass import Transformation
from tempo.utils import logger

log = logger.get_logger(__name__)

# NOTE: according to benchmarks, the only desirable case for CPU tensors is JAX index_slice start
# param.
# This is what we were doing in any case, so we're good.

#     Shape |  Idx Size | Framework |       Operation |  Index Loc |  Avg Time (ms)
# -------------------------------------------------------------------------------------
#      1000 |        10 |  PyTorch |    index_select |        GPU | 14.2608642578125
#      1000 |        10 |  PyTorch |    index_select |        CPU |          ERROR
#      1000 |        10 |  PyTorch |          narrow |        GPU | 0.064849853515625
#      1000 |        10 |      JAX |            take |        CPU | 13.167428970336914
#      1000 |        10 |      JAX |            take |        GPU | 8.981180191040039
#      1000 |        10 |      JAX |   dynamic_slice |        CPU | 5.846595764160156
#      1000 |        10 |      JAX |   dynamic_slice |        GPU | 20.350313186645508
#      1000 |       100 |  PyTorch |    index_select |        GPU | 0.1269817352294922
#      1000 |       100 |  PyTorch |    index_select |        CPU |          ERROR
#      1000 |       100 |  PyTorch |          narrow |        GPU | 0.04749298095703125
#      1000 |       100 |      JAX |            take |        CPU | 8.687067031860352
#      1000 |       100 |      JAX |            take |        GPU | 6.198644638061523
#      1000 |       100 |      JAX |   dynamic_slice |        CPU | 5.095386505126953
#      1000 |       100 |      JAX |   dynamic_slice |        GPU | 3.4094810485839844
#     10000 |        10 |  PyTorch |    index_select |        GPU | 0.058746337890625
#     10000 |        10 |  PyTorch |    index_select |        CPU |          ERROR
#     10000 |        10 |  PyTorch |          narrow |        GPU | 0.03757476806640625
#     10000 |        10 |      JAX |            take |        CPU | 8.440065383911133
#     10000 |        10 |      JAX |            take |        GPU | 6.138420104980469
#     10000 |        10 |      JAX |   dynamic_slice |        CPU | 5.164480209350586
#     10000 |        10 |      JAX |   dynamic_slice |        GPU | 3.259563446044922
#     10000 |       100 |  PyTorch |    index_select |        GPU | 0.057315826416015625
#     10000 |       100 |  PyTorch |    index_select |        CPU |          ERROR
#     10000 |       100 |  PyTorch |          narrow |        GPU | 0.037479400634765625
#     10000 |       100 |      JAX |            take |        CPU | 8.835506439208984
#     10000 |       100 |      JAX |            take |        GPU | 6.126928329467773
#     10000 |       100 |      JAX |   dynamic_slice |        CPU | 5.810642242431641
#     10000 |       100 |      JAX |   dynamic_slice |        GPU | 3.276205062866211
#     10000 |      1000 |  PyTorch |    index_select |        GPU | 0.057697296142578125
#     10000 |      1000 |  PyTorch |    index_select |        CPU |          ERROR
#     10000 |      1000 |  PyTorch |          narrow |        GPU | 0.036907196044921875
#     10000 |      1000 |      JAX |            take |        CPU | 8.675622940063477
#     10000 |      1000 |      JAX |            take |        GPU | 6.239986419677734
#     10000 |      1000 |      JAX |   dynamic_slice |        CPU | 5.127954483032227
#     10000 |      1000 |      JAX |   dynamic_slice |        GPU | 3.3803939819335938
#    100000 |        10 |  PyTorch |    index_select |        GPU | 0.05517005920410156
#    100000 |        10 |  PyTorch |    index_select |        CPU |          ERROR
#    100000 |        10 |  PyTorch |          narrow |        GPU | 0.03528594970703125
#    100000 |        10 |      JAX |            take |        CPU | 8.115577697753906
#    100000 |        10 |      JAX |            take |        GPU | 6.052350997924805
#    100000 |        10 |      JAX |   dynamic_slice |        CPU | 4.8564910888671875
#    100000 |        10 |      JAX |   dynamic_slice |        GPU | 3.1997203826904297
#    100000 |       100 |  PyTorch |    index_select |        GPU | 0.0553131103515625
#    100000 |       100 |  PyTorch |    index_select |        CPU |          ERROR
#    100000 |       100 |  PyTorch |          narrow |        GPU | 0.03619194030761719
#    100000 |       100 |      JAX |            take |        CPU | 7.958316802978516
#    100000 |       100 |      JAX |            take |        GPU | 5.935239791870117
#    100000 |       100 |      JAX |   dynamic_slice |        CPU | 4.863452911376953
#    100000 |       100 |      JAX |   dynamic_slice |        GPU | 3.1847000122070312
#    100000 |      1000 |  PyTorch |    index_select |        GPU | 0.05478858947753906
#    100000 |      1000 |  PyTorch |    index_select |        CPU |          ERROR
#    100000 |      1000 |  PyTorch |          narrow |        GPU | 0.0370025634765625
#    100000 |      1000 |      JAX |            take |        CPU | 8.299112319946289
#    100000 |      1000 |      JAX |            take |        GPU | 17.80219078063965
#    100000 |      1000 |      JAX |   dynamic_slice |        CPU | 5.19719123840332
#    100000 |      1000 |      JAX |   dynamic_slice |        GPU | 3.356647491455078
#
#
#
#
#     Shape |  Idx Size | Framework |       Operation |  Index Loc |  Avg Time (ms)
# -------------------------------------------------------------------------------------
#      1000 |        10 |  PyTorch |    index_select |        GPU | 0.05421638488769531
#      1000 |        10 |  PyTorch |    index_select |        CPU |          ERROR
#      1000 |        10 |  PyTorch |          narrow |        GPU | 0.03628730773925781
#      1000 |        10 |      JAX |            take |        CPU | 0.06041526794433594
#      1000 |        10 |      JAX |            take |        GPU | 0.043773651123046875
#      1000 |        10 |      JAX |   dynamic_slice |        CPU | 0.2277374267578125
#      1000 |        10 |      JAX |   dynamic_slice |        GPU | 0.43272972106933594
#      1000 |       100 |  PyTorch |    index_select |        GPU | 0.04096031188964844
#      1000 |       100 |  PyTorch |    index_select |        CPU |          ERROR
#      1000 |       100 |  PyTorch |          narrow |        GPU | 0.03151893615722656
#      1000 |       100 |      JAX |            take |        CPU | 0.06422996520996094
#      1000 |       100 |      JAX |            take |        GPU | 0.04782676696777344
#      1000 |       100 |      JAX |   dynamic_slice |        CPU | 0.22220611572265625
#      1000 |       100 |      JAX |   dynamic_slice |        GPU | 0.40907859802246094
#     10000 |        10 |  PyTorch |    index_select |        GPU | 0.040340423583984375
#     10000 |        10 |  PyTorch |    index_select |        CPU |          ERROR
#     10000 |        10 |  PyTorch |          narrow |        GPU | 0.031375885009765625
#     10000 |        10 |      JAX |            take |        CPU | 0.0598907470703125
#     10000 |        10 |      JAX |            take |        GPU | 0.05307197570800781
#     10000 |        10 |      JAX |   dynamic_slice |        CPU | 0.2228260040283203
#     10000 |        10 |      JAX |   dynamic_slice |        GPU | 0.42133331298828125
#     10000 |       100 |  PyTorch |    index_select |        GPU | 0.04048347473144531
#     10000 |       100 |  PyTorch |    index_select |        CPU |          ERROR
#     10000 |       100 |  PyTorch |          narrow |        GPU | 0.031375885009765625
#     10000 |       100 |      JAX |            take |        CPU | 0.05888938903808594
#     10000 |       100 |      JAX |            take |        GPU | 0.053882598876953125
#     10000 |       100 |      JAX |   dynamic_slice |        CPU | 0.228118896484375
#     10000 |       100 |      JAX |   dynamic_slice |        GPU | 0.39696693420410156
#     10000 |      1000 |  PyTorch |    index_select |        GPU | 0.04086494445800781
#     10000 |      1000 |  PyTorch |    index_select |        CPU |          ERROR
#     10000 |      1000 |  PyTorch |          narrow |        GPU | 0.03180503845214844
#     10000 |      1000 |      JAX |            take |        CPU | 0.054645538330078125
#     10000 |      1000 |      JAX |            take |        GPU | 0.04534721374511719
#     10000 |      1000 |      JAX |   dynamic_slice |        CPU | 0.21514892578125
#     10000 |      1000 |      JAX |   dynamic_slice |        GPU | 0.3948211669921875
#    100000 |        10 |  PyTorch |    index_select |        GPU | 0.04038810729980469
#    100000 |        10 |  PyTorch |    index_select |        CPU |          ERROR
#    100000 |        10 |  PyTorch |          narrow |        GPU | 0.03142356872558594
#    100000 |        10 |      JAX |            take |        CPU | 0.051212310791015625
#    100000 |        10 |      JAX |            take |        GPU | 0.0415802001953125
#    100000 |        10 |      JAX |   dynamic_slice |        CPU | 0.2166271209716797
#    100000 |        10 |      JAX |   dynamic_slice |        GPU | 0.3921985626220703
#    100000 |       100 |  PyTorch |    index_select |        GPU | 0.04076957702636719
#    100000 |       100 |  PyTorch |    index_select |        CPU |          ERROR
#    100000 |       100 |  PyTorch |          narrow |        GPU | 0.033855438232421875
#    100000 |       100 |      JAX |            take |        CPU | 0.0598907470703125
#    100000 |       100 |      JAX |            take |        GPU | 0.053882598876953125
#    100000 |       100 |      JAX |   dynamic_slice |        CPU | 0.22373199462890625
#    100000 |       100 |      JAX |   dynamic_slice |        GPU | 0.4123687744140625
#    100000 |      1000 |  PyTorch |    index_select |        GPU | 0.040912628173828125
#    100000 |      1000 |  PyTorch |    index_select |        CPU |          ERROR
#    100000 |      1000 |  PyTorch |          narrow |        GPU | 0.03190040588378906
#    100000 |      1000 |      JAX |            take |        CPU | 0.05135536193847656
#    100000 |      1000 |      JAX |            take |        GPU | 0.045871734619140625
#    100000 |      1000 |      JAX |   dynamic_slice |        CPU | 0.2171039581298828
#    100000 |      1000 |      JAX |   dynamic_slice |        GPU | 0.3917694091796875


class AnalyseDeviceAssignment(Transformation):
    def _run(self) -> tuple[PDG, bool]:
        dg = self.ctx.dg
        analysis_ctx = self.ctx.analysis_ctx
        device_assignment: dict[OpId, DeviceGroup] = {}

        default_dev = device.from_(self.ctx.exec_cfg.dev)
        bend = self.ctx.exec_cfg.get_canonical_backend_name()

        place_indexes_on_cpu = bend == DLBackendName.JAX

        for op in dg.nodes:
            if op.op_id not in device_assignment:
                device_assignment[op.op_id] = default_dev

            if place_indexes_on_cpu:
                if isinstance(op, top.IndexSliceOp):
                    (
                        _,
                        (slice_index_op, _),
                    ) = dg.get_flat_direct_dependencies(op)

                    to_mark = {slice_index_op}
                    for dep, _ in dg.get_flat_recursive_dependencies(slice_index_op):
                        to_mark.add(dep)

                    for dep in to_mark:
                        device_assignment[dep.op_id] = device.cpu
                if op.op_id not in device_assignment:
                    device_assignment[op.op_id] = default_dev

        inverse_assignment: dict[DeviceGroup, int] = {}
        for _, dev in device_assignment.items():
            inverse_assignment[dev] = inverse_assignment.get(dev, 0) + 1

        percentage_map: dict[DeviceGroup, float] = {
            dev: (count / len(device_assignment)) * 100 for dev, count in inverse_assignment.items()
        }

        log.info(
            "Device assignment map: %s",
            percentage_map,
        )
        analysis_ctx._device_assignment = device_assignment

        return dg, True
