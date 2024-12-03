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
    sphere { m*<-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 1 }        
    sphere {  m*<0.3191615167050949,0.17542330373691245,5.605136123945914>, 1 }
    sphere {  m*<2.5317154685266234,-0.0017148804498386266,-2.104079700065739>, 1 }
    sphere {  m*<-1.8246082853725238,2.224725088582386,-1.848815940030526>, 1}
    sphere { m*<-1.556821064334692,-2.662966853821511,-1.659269654867953>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3191615167050949,0.17542330373691245,5.605136123945914>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5 }
    cylinder { m*<2.5317154685266234,-0.0017148804498386266,-2.104079700065739>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5}
    cylinder { m*<-1.8246082853725238,2.224725088582386,-1.848815940030526>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5 }
    cylinder {  m*<-1.556821064334692,-2.662966853821511,-1.659269654867953>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5}

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
    sphere { m*<-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 1 }        
    sphere {  m*<0.3191615167050949,0.17542330373691245,5.605136123945914>, 1 }
    sphere {  m*<2.5317154685266234,-0.0017148804498386266,-2.104079700065739>, 1 }
    sphere {  m*<-1.8246082853725238,2.224725088582386,-1.848815940030526>, 1}
    sphere { m*<-1.556821064334692,-2.662966853821511,-1.659269654867953>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3191615167050949,0.17542330373691245,5.605136123945914>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5 }
    cylinder { m*<2.5317154685266234,-0.0017148804498386266,-2.104079700065739>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5}
    cylinder { m*<-1.8246082853725238,2.224725088582386,-1.848815940030526>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5 }
    cylinder {  m*<-1.556821064334692,-2.662966853821511,-1.659269654867953>, <-0.2029929254796338,-0.10374885583621285,-0.8748701746145582>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    