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
    sphere { m*<-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 1 }        
    sphere {  m*<0.10255824733818641,0.07538494220632308,2.779445514761235>, 1 }
    sphere {  m*<2.596531536602756,0.04870883941237225,-1.4373187818104998>, 1 }
    sphere {  m*<-1.7597922172963987,2.2751488084446003,-1.1820550217752857>, 1}
    sphere { m*<-1.567819372034437,-2.7558593418464614,-1.0364351075068918>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10255824733818641,0.07538494220632308,2.779445514761235>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5 }
    cylinder { m*<2.596531536602756,0.04870883941237225,-1.4373187818104998>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5}
    cylinder { m*<-1.7597922172963987,2.2751488084446003,-1.1820550217752857>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5 }
    cylinder {  m*<-1.567819372034437,-2.7558593418464614,-1.0364351075068918>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5}

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
    sphere { m*<-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 1 }        
    sphere {  m*<0.10255824733818641,0.07538494220632308,2.779445514761235>, 1 }
    sphere {  m*<2.596531536602756,0.04870883941237225,-1.4373187818104998>, 1 }
    sphere {  m*<-1.7597922172963987,2.2751488084446003,-1.1820550217752857>, 1}
    sphere { m*<-1.567819372034437,-2.7558593418464614,-1.0364351075068918>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10255824733818641,0.07538494220632308,2.779445514761235>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5 }
    cylinder { m*<2.596531536602756,0.04870883941237225,-1.4373187818104998>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5}
    cylinder { m*<-1.7597922172963987,2.2751488084446003,-1.1820550217752857>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5 }
    cylinder {  m*<-1.567819372034437,-2.7558593418464614,-1.0364351075068918>, <-0.13817685740350505,-0.053325135974002125,-0.2081092563593147>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    