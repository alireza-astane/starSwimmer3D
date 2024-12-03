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
    sphere { m*<1.0695842858249087,0.35898636433743913,0.4982766913891356>, 1 }        
    sphere {  m*<1.3136337950458155,0.38722338922452365,3.4881989848242085>, 1 }
    sphere {  m*<3.8068809841083495,0.38722338922452354,-0.7290832236664098>, 1 }
    sphere {  m*<-3.049816475086901,6.897846500173174,-1.9373802885750877>, 1}
    sphere { m*<-3.7845280789850233,-7.899012511363282,-2.3711130540930165>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3136337950458155,0.38722338922452365,3.4881989848242085>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5 }
    cylinder { m*<3.8068809841083495,0.38722338922452354,-0.7290832236664098>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5}
    cylinder { m*<-3.049816475086901,6.897846500173174,-1.9373802885750877>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5 }
    cylinder {  m*<-3.7845280789850233,-7.899012511363282,-2.3711130540930165>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5}

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
    sphere { m*<1.0695842858249087,0.35898636433743913,0.4982766913891356>, 1 }        
    sphere {  m*<1.3136337950458155,0.38722338922452365,3.4881989848242085>, 1 }
    sphere {  m*<3.8068809841083495,0.38722338922452354,-0.7290832236664098>, 1 }
    sphere {  m*<-3.049816475086901,6.897846500173174,-1.9373802885750877>, 1}
    sphere { m*<-3.7845280789850233,-7.899012511363282,-2.3711130540930165>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3136337950458155,0.38722338922452365,3.4881989848242085>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5 }
    cylinder { m*<3.8068809841083495,0.38722338922452354,-0.7290832236664098>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5}
    cylinder { m*<-3.049816475086901,6.897846500173174,-1.9373802885750877>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5 }
    cylinder {  m*<-3.7845280789850233,-7.899012511363282,-2.3711130540930165>, <1.0695842858249087,0.35898636433743913,0.4982766913891356>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    