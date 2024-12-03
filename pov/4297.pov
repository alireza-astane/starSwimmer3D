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
    sphere { m*<-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 1 }        
    sphere {  m*<0.2287508137573659,0.12708482540311644,4.4831272368678325>, 1 }
    sphere {  m*<2.5553329601915418,0.010912314914445984,-1.810983477223056>, 1 }
    sphere {  m*<-1.800990793707605,2.237352283946671,-1.5557197171878425>, 1}
    sphere { m*<-1.5332035726697733,-2.6503396584572263,-1.3661734320252699>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2287508137573659,0.12708482540311644,4.4831272368678325>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5 }
    cylinder { m*<2.5553329601915418,0.010912314914445984,-1.810983477223056>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5}
    cylinder { m*<-1.800990793707605,2.237352283946671,-1.5557197171878425>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5 }
    cylinder {  m*<-1.5332035726697733,-2.6503396584572263,-1.3661734320252699>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5}

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
    sphere { m*<-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 1 }        
    sphere {  m*<0.2287508137573659,0.12708482540311644,4.4831272368678325>, 1 }
    sphere {  m*<2.5553329601915418,0.010912314914445984,-1.810983477223056>, 1 }
    sphere {  m*<-1.800990793707605,2.237352283946671,-1.5557197171878425>, 1}
    sphere { m*<-1.5332035726697733,-2.6503396584572263,-1.3661734320252699>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2287508137573659,0.12708482540311644,4.4831272368678325>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5 }
    cylinder { m*<2.5553329601915418,0.010912314914445984,-1.810983477223056>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5}
    cylinder { m*<-1.800990793707605,2.237352283946671,-1.5557197171878425>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5 }
    cylinder {  m*<-1.5332035726697733,-2.6503396584572263,-1.3661734320252699>, <-0.17937543381471505,-0.09112166047192818,-0.5817739517718735>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    