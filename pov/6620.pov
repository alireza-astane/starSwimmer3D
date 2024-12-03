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
    sphere { m*<-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 1 }        
    sphere {  m*<0.2993621512210669,-0.11277257630331544,9.073216045548385>, 1 }
    sphere {  m*<7.654713589221033,-0.20169285229767234,-5.506277244496957>, 1 }
    sphere {  m*<-5.541086932015626,4.5712764081853,-3.047526942986824>, 1}
    sphere { m*<-2.4140153352240055,-3.50626243102896,-1.4204899808260447>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2993621512210669,-0.11277257630331544,9.073216045548385>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5 }
    cylinder { m*<7.654713589221033,-0.20169285229767234,-5.506277244496957>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5}
    cylinder { m*<-5.541086932015626,4.5712764081853,-3.047526942986824>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5 }
    cylinder {  m*<-2.4140153352240055,-3.50626243102896,-1.4204899808260447>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5}

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
    sphere { m*<-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 1 }        
    sphere {  m*<0.2993621512210669,-0.11277257630331544,9.073216045548385>, 1 }
    sphere {  m*<7.654713589221033,-0.20169285229767234,-5.506277244496957>, 1 }
    sphere {  m*<-5.541086932015626,4.5712764081853,-3.047526942986824>, 1}
    sphere { m*<-2.4140153352240055,-3.50626243102896,-1.4204899808260447>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2993621512210669,-0.11277257630331544,9.073216045548385>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5 }
    cylinder { m*<7.654713589221033,-0.20169285229767234,-5.506277244496957>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5}
    cylinder { m*<-5.541086932015626,4.5712764081853,-3.047526942986824>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5 }
    cylinder {  m*<-2.4140153352240055,-3.50626243102896,-1.4204899808260447>, <-1.1405246154914228,-0.8634915795582343,-0.7941360964827961>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    