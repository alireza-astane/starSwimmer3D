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
    sphere { m*<-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 1 }        
    sphere {  m*<0.21066467195226485,0.11741499046893511,4.258675803833293>, 1 }
    sphere {  m*<2.559715428162332,0.013255420610428365,-1.7565964649530101>, 1 }
    sphere {  m*<-1.7966083257368148,2.2396953896426535,-1.5013327049177967>, 1}
    sphere { m*<-1.528821104698983,-2.647996552761244,-1.311786419755224>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21066467195226485,0.11741499046893511,4.258675803833293>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5 }
    cylinder { m*<2.559715428162332,0.013255420610428365,-1.7565964649530101>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5}
    cylinder { m*<-1.7966083257368148,2.2396953896426535,-1.5013327049177967>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5 }
    cylinder {  m*<-1.528821104698983,-2.647996552761244,-1.311786419755224>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5}

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
    sphere { m*<-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 1 }        
    sphere {  m*<0.21066467195226485,0.11741499046893511,4.258675803833293>, 1 }
    sphere {  m*<2.559715428162332,0.013255420610428365,-1.7565964649530101>, 1 }
    sphere {  m*<-1.7966083257368148,2.2396953896426535,-1.5013327049177967>, 1}
    sphere { m*<-1.528821104698983,-2.647996552761244,-1.311786419755224>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.21066467195226485,0.11741499046893511,4.258675803833293>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5 }
    cylinder { m*<2.559715428162332,0.013255420610428365,-1.7565964649530101>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5}
    cylinder { m*<-1.7966083257368148,2.2396953896426535,-1.5013327049177967>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5 }
    cylinder {  m*<-1.528821104698983,-2.647996552761244,-1.311786419755224>, <-0.17499296584392482,-0.08877855477594578,-0.5273869395018271>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    