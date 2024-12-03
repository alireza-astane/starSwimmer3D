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
    sphere { m*<0.7552401147178381,0.8454156782207521,0.31241655538167257>, 1 }        
    sphere {  m*<0.9980533915864457,0.921377994280504,3.3016059689456716>, 1 }
    sphere {  m*<3.4913005806489807,0.9213779942805038,-0.9156762395449427>, 1 }
    sphere {  m*<-2.0376232867222117,5.031193302646503,-1.338894993142634>, 1}
    sphere { m*<-3.901099501452554,-7.5649042241284326,-2.4400439582705573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9980533915864457,0.921377994280504,3.3016059689456716>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5 }
    cylinder { m*<3.4913005806489807,0.9213779942805038,-0.9156762395449427>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5}
    cylinder { m*<-2.0376232867222117,5.031193302646503,-1.338894993142634>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5 }
    cylinder {  m*<-3.901099501452554,-7.5649042241284326,-2.4400439582705573>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5}

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
    sphere { m*<0.7552401147178381,0.8454156782207521,0.31241655538167257>, 1 }        
    sphere {  m*<0.9980533915864457,0.921377994280504,3.3016059689456716>, 1 }
    sphere {  m*<3.4913005806489807,0.9213779942805038,-0.9156762395449427>, 1 }
    sphere {  m*<-2.0376232867222117,5.031193302646503,-1.338894993142634>, 1}
    sphere { m*<-3.901099501452554,-7.5649042241284326,-2.4400439582705573>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9980533915864457,0.921377994280504,3.3016059689456716>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5 }
    cylinder { m*<3.4913005806489807,0.9213779942805038,-0.9156762395449427>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5}
    cylinder { m*<-2.0376232867222117,5.031193302646503,-1.338894993142634>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5 }
    cylinder {  m*<-3.901099501452554,-7.5649042241284326,-2.4400439582705573>, <0.7552401147178381,0.8454156782207521,0.31241655538167257>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    