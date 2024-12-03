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
    sphere { m*<-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 1 }        
    sphere {  m*<0.8673221563266956,0.26758728779125507,9.344173386009356>, 1 }
    sphere {  m*<8.235109354649502,-0.017504963001006013,-5.226504043064573>, 1 }
    sphere {  m*<-6.660853839039495,6.505576410619635,-3.7356971398829657>, 1}
    sphere { m*<-3.3354243758409656,-6.784443255220065,-1.7941575061987578>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8673221563266956,0.26758728779125507,9.344173386009356>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5 }
    cylinder { m*<8.235109354649502,-0.017504963001006013,-5.226504043064573>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5}
    cylinder { m*<-6.660853839039495,6.505576410619635,-3.7356971398829657>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5 }
    cylinder {  m*<-3.3354243758409656,-6.784443255220065,-1.7941575061987578>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5}

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
    sphere { m*<-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 1 }        
    sphere {  m*<0.8673221563266956,0.26758728779125507,9.344173386009356>, 1 }
    sphere {  m*<8.235109354649502,-0.017504963001006013,-5.226504043064573>, 1 }
    sphere {  m*<-6.660853839039495,6.505576410619635,-3.7356971398829657>, 1}
    sphere { m*<-3.3354243758409656,-6.784443255220065,-1.7941575061987578>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8673221563266956,0.26758728779125507,9.344173386009356>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5 }
    cylinder { m*<8.235109354649502,-0.017504963001006013,-5.226504043064573>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5}
    cylinder { m*<-6.660853839039495,6.505576410619635,-3.7356971398829657>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5 }
    cylinder {  m*<-3.3354243758409656,-6.784443255220065,-1.7941575061987578>, <-0.5518453378734658,-0.722351626088662,-0.5051167110257901>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    