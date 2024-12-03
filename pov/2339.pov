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
    sphere { m*<1.0156164704295132,0.4471152194552357,0.46636731330798586>, 1 }        
    sphere {  m*<1.2595420163921545,0.4831173658586675,3.4562159909308168>, 1 }
    sphere {  m*<3.7527892054546905,0.4831173658586673,-0.761066217559802>, 1 }
    sphere {  m*<-2.881952383211924,6.575952156044449,-1.8381257980467025>, 1}
    sphere { m*<-3.806574616765223,-7.836089650873106,-2.3841495808908117>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2595420163921545,0.4831173658586675,3.4562159909308168>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5 }
    cylinder { m*<3.7527892054546905,0.4831173658586673,-0.761066217559802>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5}
    cylinder { m*<-2.881952383211924,6.575952156044449,-1.8381257980467025>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5 }
    cylinder {  m*<-3.806574616765223,-7.836089650873106,-2.3841495808908117>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5}

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
    sphere { m*<1.0156164704295132,0.4471152194552357,0.46636731330798586>, 1 }        
    sphere {  m*<1.2595420163921545,0.4831173658586675,3.4562159909308168>, 1 }
    sphere {  m*<3.7527892054546905,0.4831173658586673,-0.761066217559802>, 1 }
    sphere {  m*<-2.881952383211924,6.575952156044449,-1.8381257980467025>, 1}
    sphere { m*<-3.806574616765223,-7.836089650873106,-2.3841495808908117>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2595420163921545,0.4831173658586675,3.4562159909308168>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5 }
    cylinder { m*<3.7527892054546905,0.4831173658586673,-0.761066217559802>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5}
    cylinder { m*<-2.881952383211924,6.575952156044449,-1.8381257980467025>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5 }
    cylinder {  m*<-3.806574616765223,-7.836089650873106,-2.3841495808908117>, <1.0156164704295132,0.4471152194552357,0.46636731330798586>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    