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
    sphere { m*<0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 1 }        
    sphere {  m*<0.6995520744672676,-1.3930747674661511e-18,3.9617063290950174>, 1 }
    sphere {  m*<7.032129685860082,2.4144730715576396e-18,-1.5322301946355286>, 1 }
    sphere {  m*<-4.206165437197252,8.164965809277259,-2.2243587868646575>, 1}
    sphere { m*<-4.206165437197252,-8.164965809277259,-2.224358786864661>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6995520744672676,-1.3930747674661511e-18,3.9617063290950174>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5 }
    cylinder { m*<7.032129685860082,2.4144730715576396e-18,-1.5322301946355286>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5}
    cylinder { m*<-4.206165437197252,8.164965809277259,-2.2243587868646575>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5 }
    cylinder {  m*<-4.206165437197252,-8.164965809277259,-2.224358786864661>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5}

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
    sphere { m*<0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 1 }        
    sphere {  m*<0.6995520744672676,-1.3930747674661511e-18,3.9617063290950174>, 1 }
    sphere {  m*<7.032129685860082,2.4144730715576396e-18,-1.5322301946355286>, 1 }
    sphere {  m*<-4.206165437197252,8.164965809277259,-2.2243587868646575>, 1}
    sphere { m*<-4.206165437197252,-8.164965809277259,-2.224358786864661>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6995520744672676,-1.3930747674661511e-18,3.9617063290950174>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5 }
    cylinder { m*<7.032129685860082,2.4144730715576396e-18,-1.5322301946355286>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5}
    cylinder { m*<-4.206165437197252,8.164965809277259,-2.2243587868646575>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5 }
    cylinder {  m*<-4.206165437197252,-8.164965809277259,-2.224358786864661>, <0.6077158037534726,-4.877621229642073e-18,0.9631091165029544>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    