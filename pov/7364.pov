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
    sphere { m*<-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 1 }        
    sphere {  m*<0.7803354533518708,0.0781472335456721,9.303890926900884>, 1 }
    sphere {  m*<8.148122651674669,-0.20694501724659076,-5.26678650217305>, 1 }
    sphere {  m*<-6.747840542014319,6.316136356374054,-3.7759795989914453>, 1}
    sphere { m*<-2.9147786931752657,-5.868359092731201,-1.5993617278595909>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7803354533518708,0.0781472335456721,9.303890926900884>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5 }
    cylinder { m*<8.148122651674669,-0.20694501724659076,-5.26678650217305>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5}
    cylinder { m*<-6.747840542014319,6.316136356374054,-3.7759795989914453>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5 }
    cylinder {  m*<-2.9147786931752657,-5.868359092731201,-1.5993617278595909>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5}

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
    sphere { m*<-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 1 }        
    sphere {  m*<0.7803354533518708,0.0781472335456721,9.303890926900884>, 1 }
    sphere {  m*<8.148122651674669,-0.20694501724659076,-5.26678650217305>, 1 }
    sphere {  m*<-6.747840542014319,6.316136356374054,-3.7759795989914453>, 1}
    sphere { m*<-2.9147786931752657,-5.868359092731201,-1.5993617278595909>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7803354533518708,0.0781472335456721,9.303890926900884>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5 }
    cylinder { m*<8.148122651674669,-0.20694501724659076,-5.26678650217305>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5}
    cylinder { m*<-6.747840542014319,6.316136356374054,-3.7759795989914453>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5 }
    cylinder {  m*<-2.9147786931752657,-5.868359092731201,-1.5993617278595909>, <-0.6388320408482917,-0.9117916803342458,-0.5453991701342691>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    