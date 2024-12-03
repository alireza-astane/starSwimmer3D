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
    sphere { m*<-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 1 }        
    sphere {  m*<-7.177936856924139e-18,-2.283062609170685e-18,8.144336770642933>, 1 }
    sphere {  m*<9.428090415820634,-2.650243150062411e-18,-2.920996562690414>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.920996562690414>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.920996562690414>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-7.177936856924139e-18,-2.283062609170685e-18,8.144336770642933>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5 }
    cylinder { m*<9.428090415820634,-2.650243150062411e-18,-2.920996562690414>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.920996562690414>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.920996562690414>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5}

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
    sphere { m*<-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 1 }        
    sphere {  m*<-7.177936856924139e-18,-2.283062609170685e-18,8.144336770642933>, 1 }
    sphere {  m*<9.428090415820634,-2.650243150062411e-18,-2.920996562690414>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.920996562690414>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.920996562690414>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-7.177936856924139e-18,-2.283062609170685e-18,8.144336770642933>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5 }
    cylinder { m*<9.428090415820634,-2.650243150062411e-18,-2.920996562690414>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.920996562690414>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.920996562690414>, <-4.89168284261138e-18,-2.178149370529817e-19,0.412336770642918>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    