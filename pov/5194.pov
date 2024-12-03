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
    sphere { m*<-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 1 }        
    sphere {  m*<0.438697632928216,0.2886941121602233,8.364885052359451>, 1 }
    sphere {  m*<3.4574986168583606,-0.0021008318284058414,-3.4233141765082657>, 1 }
    sphere {  m*<-2.1306354764128197,2.1814293704761516,-2.535508407232279>, 1}
    sphere { m*<-1.8628482553749879,-2.7062625719277458,-2.3459621220697087>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.438697632928216,0.2886941121602233,8.364885052359451>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5 }
    cylinder { m*<3.4574986168583606,-0.0021008318284058414,-3.4233141765082657>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5}
    cylinder { m*<-2.1306354764128197,2.1814293704761516,-2.535508407232279>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5 }
    cylinder {  m*<-1.8628482553749879,-2.7062625719277458,-2.3459621220697087>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5}

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
    sphere { m*<-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 1 }        
    sphere {  m*<0.438697632928216,0.2886941121602233,8.364885052359451>, 1 }
    sphere {  m*<3.4574986168583606,-0.0021008318284058414,-3.4233141765082657>, 1 }
    sphere {  m*<-2.1306354764128197,2.1814293704761516,-2.535508407232279>, 1}
    sphere { m*<-1.8628482553749879,-2.7062625719277458,-2.3459621220697087>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.438697632928216,0.2886941121602233,8.364885052359451>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5 }
    cylinder { m*<3.4574986168583606,-0.0021008318284058414,-3.4233141765082657>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5}
    cylinder { m*<-2.1306354764128197,2.1814293704761516,-2.535508407232279>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5 }
    cylinder {  m*<-1.8628482553749879,-2.7062625719277458,-2.3459621220697087>, <-0.49731399838172263,-0.1471815598363687,-1.5816595829680347>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    