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
    sphere { m*<2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 1 }        
    sphere {  m*<5.81998768047275e-19,-3.8070310100253445e-18,5.579663250314341>, 1 }
    sphere {  m*<9.428090415820634,-1.926012236234424e-19,-2.3916700830190156>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.3916700830190156>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.3916700830190156>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<5.81998768047275e-19,-3.8070310100253445e-18,5.579663250314341>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5 }
    cylinder { m*<9.428090415820634,-1.926012236234424e-19,-2.3916700830190156>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.3916700830190156>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.3916700830190156>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5}

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
    sphere { m*<2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 1 }        
    sphere {  m*<5.81998768047275e-19,-3.8070310100253445e-18,5.579663250314341>, 1 }
    sphere {  m*<9.428090415820634,-1.926012236234424e-19,-2.3916700830190156>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.3916700830190156>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.3916700830190156>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<5.81998768047275e-19,-3.8070310100253445e-18,5.579663250314341>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5 }
    cylinder { m*<9.428090415820634,-1.926012236234424e-19,-2.3916700830190156>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.3916700830190156>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.3916700830190156>, <2.9578788304561543e-18,-5.172727427534264e-18,0.9416632503143163>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    