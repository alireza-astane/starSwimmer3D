#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 1 }        
    sphere {  m*<-3.4510511776835154e-18,-5.848886008605313e-18,7.203944626676156>, 1 }
    sphere {  m*<9.428090415820634,-5.904617774203542e-19,-2.720388706657197>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.720388706657197>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.720388706657197>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.4510511776835154e-18,-5.848886008605313e-18,7.203944626676156>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5 }
    cylinder { m*<9.428090415820634,-5.904617774203542e-19,-2.720388706657197>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.720388706657197>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.720388706657197>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 1 }        
    sphere {  m*<-3.4510511776835154e-18,-5.848886008605313e-18,7.203944626676156>, 1 }
    sphere {  m*<9.428090415820634,-5.904617774203542e-19,-2.720388706657197>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.720388706657197>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.720388706657197>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-3.4510511776835154e-18,-5.848886008605313e-18,7.203944626676156>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5 }
    cylinder { m*<9.428090415820634,-5.904617774203542e-19,-2.720388706657197>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.720388706657197>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.720388706657197>, <-3.0118549106456815e-18,-2.8749554975990578e-18,0.6129446266761365>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    