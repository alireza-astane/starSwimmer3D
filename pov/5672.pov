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
    sphere { m*<-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 1 }        
    sphere {  m*<0.13398470301537718,0.28232368500197697,8.631096593237602>, 1 }
    sphere {  m*<5.739451345780647,0.07089201000559972,-4.763135612238922>, 1 }
    sphere {  m*<-2.7960654575996315,2.1587545899204414,-2.180741142061611>, 1}
    sphere { m*<-2.5282782365618,-2.728937352483456,-1.9911948568990407>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13398470301537718,0.28232368500197697,8.631096593237602>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5 }
    cylinder { m*<5.739451345780647,0.07089201000559972,-4.763135612238922>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5}
    cylinder { m*<-2.7960654575996315,2.1587545899204414,-2.180741142061611>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5 }
    cylinder {  m*<-2.5282782365618,-2.728937352483456,-1.9911948568990407>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5}

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
    sphere { m*<-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 1 }        
    sphere {  m*<0.13398470301537718,0.28232368500197697,8.631096593237602>, 1 }
    sphere {  m*<5.739451345780647,0.07089201000559972,-4.763135612238922>, 1 }
    sphere {  m*<-2.7960654575996315,2.1587545899204414,-2.180741142061611>, 1}
    sphere { m*<-2.5282782365618,-2.728937352483456,-1.9911948568990407>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13398470301537718,0.28232368500197697,8.631096593237602>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5 }
    cylinder { m*<5.739451345780647,0.07089201000559972,-4.763135612238922>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5}
    cylinder { m*<-2.7960654575996315,2.1587545899204414,-2.180741142061611>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5 }
    cylinder {  m*<-2.5282782365618,-2.728937352483456,-1.9911948568990407>, <-1.1347017377032733,-0.1702889828194637,-1.2777367008133855>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    