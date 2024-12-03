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
    sphere { m*<-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 1 }        
    sphere {  m*<0.5185762726890852,0.29045587026596603,8.29747267970298>, 1 }
    sphere {  m*<2.6539274759848737,-0.029622484626886394,-2.9985094697297576>, 1 }
    sphere {  m*<-1.9310606217418014,2.188710204944172,-2.629492214194795>, 1}
    sphere { m*<-1.6632734007039696,-2.6989817374597256,-2.4399459290322247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5185762726890852,0.29045587026596603,8.29747267970298>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5 }
    cylinder { m*<2.6539274759848737,-0.029622484626886394,-2.9985094697297576>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5}
    cylinder { m*<-1.9310606217418014,2.188710204944172,-2.629492214194795>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5 }
    cylinder {  m*<-1.6632734007039696,-2.6989817374597256,-2.4399459290322247>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5}

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
    sphere { m*<-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 1 }        
    sphere {  m*<0.5185762726890852,0.29045587026596603,8.29747267970298>, 1 }
    sphere {  m*<2.6539274759848737,-0.029622484626886394,-2.9985094697297576>, 1 }
    sphere {  m*<-1.9310606217418014,2.188710204944172,-2.629492214194795>, 1}
    sphere { m*<-1.6632734007039696,-2.6989817374597256,-2.4399459290322247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5185762726890852,0.29045587026596603,8.29747267970298>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5 }
    cylinder { m*<2.6539274759848737,-0.029622484626886394,-2.9985094697297576>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5}
    cylinder { m*<-1.9310606217418014,2.188710204944172,-2.629492214194795>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5 }
    cylinder {  m*<-1.6632734007039696,-2.6989817374597256,-2.4399459290322247>, <-0.30736763373986414,-0.13978631794830462,-1.6590684613035196>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    